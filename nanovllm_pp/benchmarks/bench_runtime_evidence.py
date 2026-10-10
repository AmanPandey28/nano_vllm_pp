"""Same-runner batching, prefix reuse, and bounded KV-pressure experiments.

Run: python -m nanovllm_pp.benchmarks.bench_runtime_evidence MODEL --output-dir DIR
"""

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from nanovllm_pp import LLM

from .evidence import environment_manifest
from .serving_harness import run_serving_workload
from .workloads import (
    BenchmarkWorkload,
    make_prefix_repetition_workload,
    make_random_workload,
)


def mixed_workload(tokenizer, num_requests, seed):
    requests = []
    for index in range(num_requests):
        prompt_len, output_len = ((32, 32), (256, 16), (512, 32))[index % 3]
        requests.extend(
            make_random_workload(
                tokenizer,
                num_prompts=1,
                input_len=prompt_len,
                output_len=output_len,
                seed=seed + index,
            ).requests
        )
    return BenchmarkWorkload(
        "mixed-32-256-512", tuple(requests), metadata={"seed": seed}
    )


def summarize_case(name, trials):
    return {
        "case": name,
        "output_tps": statistics.median(
            t["output_token_throughput_tps"] for t in trials
        ),
        "ttft_p50_ms": statistics.median(
            t["latency_ms"]["ttft"]["p50"] for t in trials
        ),
        "ttft_p95_ms": statistics.median(
            t["latency_ms"]["ttft"]["p95"] for t in trials
        ),
        "tpot_p95_ms": statistics.median(
            t["latency_ms"]["tpot"]["p95"] for t in trials
        ),
        "e2e_p95_ms": statistics.median(t["latency_ms"]["e2e"]["p95"] for t in trials),
        "cached_tokens": statistics.median(t["cached_prompt_tokens"] for t in trials),
        "prefill_tokens_computed": statistics.median(
            t["prefill_tokens_computed"] for t in trials
        ),
        "prefix_token_hit_fraction": statistics.median(
            t["prefix_cache_hit_fraction"] for t in trials
        ),
        "max_kv_utilization": max(t["max_kv_utilization"] for t in trials),
        "max_waiting_requests": max(t["max_waiting_requests"] for t in trials),
        "trial_output_tps": [t["output_token_throughput_tps"] for t in trials],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--output-dir", default="runs/runtime_evidence")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-prompts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.repeats < 1 or args.num_prompts < 3:
        parser.error("repeats must be positive; num-prompts must be at least 3")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    engine = LLM(
        args.model_path,
        device="cuda",
        dtype="bfloat16",
        model_backend="paged",
        block_size=16,
        kv_cache_num_blocks=512,
        max_model_len=2048,
        max_num_batched_tokens=1024,
        max_prefill_chunk_size=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    )
    manifest = environment_manifest(args.model_path)
    cases = {}
    rows = []

    def measure(
        name, workload, *, slots, concurrency, rate=math.inf, prefix=False, state="warm"
    ):
        engine.engine_config.max_num_seqs = slots
        engine.block_manager.enable_prefix_caching = prefix
        engine.engine_config.enable_prefix_caching = prefix
        trials = []
        for _ in range(args.repeats):
            # Each trial starts with a fresh prefix index. Warmup is excluded
            # from timing; prefix workloads explicitly seed a warm common page.
            engine.block_manager.hash_to_block_id.clear()
            result = run_serving_workload(
                engine,
                workload,
                request_rate=rate,
                max_concurrency=concurrency,
                seed=args.seed,
                save_detailed=True,
                prefix_cache_state=state,
            )
            engine.block_manager.assert_invariants([])
            trials.append(result)
        case = {
            "engine_config": asdict(engine.engine_config),
            "request_rate": "inf" if math.isinf(rate) else rate,
            "client_concurrency": concurrency,
            "workload": asdict(workload),
            "trials": trials,
            "summary": summarize_case(name, trials),
        }
        cases[name] = case
        rows.append(case["summary"])
        (directory / f"{name}.json").write_text(
            json.dumps({"manifest": manifest, **case}, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = case["summary"]
        print(
            f"{name}: {summary['output_tps']:.1f} output tok/s; "
            f"TTFT p50/p95 {summary['ttft_p50_ms']:.1f}/{summary['ttft_p95_ms']:.1f} ms; "
            f"reused {summary['cached_tokens']} tokens",
            flush=True,
        )
        return case

    mixed = mixed_workload(engine.tokenizer, args.num_prompts, args.seed)
    for slots in (1, 4, 8):
        measure(f"mixed_batch{slots}", mixed, slots=slots, concurrency=args.num_prompts)
    measure(
        "mixed_batch8_rate4", mixed, slots=8, concurrency=args.num_prompts, rate=4.0
    )
    for prefix_len in (64, 256, 512):
        workload = make_prefix_repetition_workload(
            engine.tokenizer,
            num_prompts=args.num_prompts,
            prefix_len=prefix_len,
            suffix_len=32,
            output_len=16,
            seed=args.seed,
        )
        baseline = measure(f"prefix{prefix_len}_off", workload, slots=4, concurrency=4)
        cached = measure(
            f"prefix{prefix_len}_warm", workload, slots=4, concurrency=4, prefix=True
        )
        for trial, reference in zip(cached["trials"], baseline["trials"]):
            trial["exact_output_matches_vs_cache_off"] = sum(
                a["generated_token_ids"] == b["generated_token_ids"]
                for a, b in zip(trial["requests"], reference["requests"])
            )
        # Keep parity evidence beside the timing rather than silently assuming it.
        (directory / f"prefix{prefix_len}_warm.json").write_text(
            json.dumps({"manifest": manifest, **cached}, indent=2) + "\n",
            encoding="utf-8",
        )
        if prefix_len == 256:
            measure(
                "prefix256_cold_start",
                workload,
                slots=4,
                concurrency=4,
                prefix=True,
                state="cold",
            )

    # Same model and request set, but fewer physical pages. Scheduling width is
    # bounded so all requests can finish without unsupported preemption.
    del (
        engine.model_runner.kv_cache,
        engine.model_runner.kv_keys,
        engine.model_runner.kv_values,
    )
    import torch

    from nanovllm_pp.engine.block_manager import BlockManager
    from nanovllm_pp.engine.scheduler import Scheduler
    from nanovllm_pp.paged.runner import PagedModelRunner

    torch.cuda.empty_cache()
    engine.num_kv_blocks = 72
    engine.engine_config.kv_cache_num_blocks = 72
    engine.block_manager = BlockManager(72, 16)
    engine.model_runner = PagedModelRunner(
        engine.model,
        engine.model_config,
        engine.engine_config,
        engine.device,
        engine.block_manager,
        72,
    )
    engine.scheduler = Scheduler(engine.engine_config, engine.block_manager)
    measure("mixed_pressure72_batch2", mixed, slots=2, concurrency=args.num_prompts)

    baseline = cases["mixed_batch1"]["trials"]
    for name in ("mixed_batch4", "mixed_batch8"):
        cases[name]["exact_output_matches_vs_sequential"] = [
            sum(
                a["generated_token_ids"] == b["generated_token_ids"]
                for a, b in zip(trial["requests"], reference["requests"])
            )
            for trial, reference in zip(cases[name]["trials"], baseline)
        ]
    (directory / "results.json").write_text(
        json.dumps({"manifest": manifest, "cases": cases}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evidence saved under {directory}")


if __name__ == "__main__":
    main()
