"""Compare the HF reference and physical paged backend on one small workload."""

import argparse
import gc
import statistics
import time

import torch

from nanovllm_pp import LLM, SamplingParams

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Machine learning is a field of",
    "Python is a programming language",
]


def _gib(num_bytes):
    return num_bytes / 1024**3


def _measure(model_path, backend, args):
    engine = LLM(
        model_path,
        device="cuda",
        dtype=args.dtype,
        model_backend=backend,
        block_size=args.block_size,
        kv_cache_num_blocks=args.kv_cache_num_blocks,
        max_num_seqs=max(8, len(DEFAULT_PROMPTS)),
        max_num_batched_tokens=256,
        max_model_len=512,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    engine.generate(["Warm the CUDA execution path"], params)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    samples = []
    token_ids = None
    for _ in range(args.trials):
        engine.enable_profiling()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(DEFAULT_PROMPTS, params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = sum(output["num_completion_tokens"] for output in outputs)
        token_ids = [
            output["token_ids"][output["num_prompt_tokens"] :] for output in outputs
        ]
        samples.append(
            {
                "elapsed_s": elapsed,
                "tokens_per_s": tokens / elapsed,
                "latency": engine._request_tracker.report(),
                "engine": engine._profiler.summary(),
                "peak_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    median_sample = sorted(samples, key=lambda sample: sample["elapsed_s"])[
        len(samples) // 2
    ]
    result = {
        "backend": backend,
        "median": median_sample,
        "median_tokens_per_s": statistics.median(
            sample["tokens_per_s"] for sample in samples
        ),
        "resident_bytes": resident,
        "token_ids": token_ids,
    }
    if backend == "paged":
        cache = engine.model_runner.kv_cache
        result["physical_kv_shape"] = tuple(cache.shape)
        result["physical_kv_bytes"] = cache.numel() * cache.element_size()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _metric(result, name):
    return result["median"]["latency"][name]["p50"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-cache-num-blocks", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if args.trials <= 0 or args.trials % 2 == 0:
        raise ValueError("--trials must be a positive odd number")

    hf = _measure(args.model_path, "hf", args)
    paged = _measure(args.model_path, "paged", args)

    print(
        f"workload: {len(DEFAULT_PROMPTS)} prompts x {args.max_tokens} forced "
        f"output tokens, {args.trials} warmed synchronized trials"
    )
    print("backend  time(s)  tok/s   TTFT p50  TPOT p50  E2E p50  resident  peak")
    for result in (hf, paged):
        median = result["median"]
        print(
            f"{result['backend']:<8} "
            f"{median['elapsed_s']:>7.3f}  "
            f"{result['median_tokens_per_s']:>6.1f}  "
            f"{_metric(result, 'ttft'):>8.1f}ms  "
            f"{_metric(result, 'tpot'):>8.1f}ms  "
            f"{_metric(result, 'e2e'):>7.1f}ms  "
            f"{_gib(result['resident_bytes']):>6.2f}GiB  "
            f"{_gib(median['peak_bytes']):>5.2f}GiB"
        )
    print(
        "paged/HF output-throughput ratio: "
        f"{paged['median_tokens_per_s'] / hf['median_tokens_per_s']:.2f}x"
    )
    print(
        "exact greedy prompt matches: "
        f"{sum(a == b for a, b in zip(hf['token_ids'], paged['token_ids']))}/"
        f"{len(DEFAULT_PROMPTS)}"
    )
    print(
        f"physical KV shape: {paged['physical_kv_shape']}; "
        f"bytes: {paged['physical_kv_bytes']:,}"
    )
    print(f"median paged engine phases: {paged['median']['engine']}")


if __name__ == "__main__":
    main()
