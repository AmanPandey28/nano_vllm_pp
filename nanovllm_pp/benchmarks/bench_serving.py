"""vLLM-shaped serving benchmark for random, ShareGPT, and prefix workloads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from nanovllm_pp import LLM

from .serving_harness import run_serving_workload
from .workloads import (
    count_prompt_tokens,
    load_sharegpt_workload,
    make_prefix_repetition_workload,
    make_random_workload,
    workload_token_stats,
)


def parse_request_rate(value: str) -> float:
    if value.lower() in {"inf", "infinity"}:
        return math.inf
    rate = float(value)
    if rate <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive or 'inf'")
    return rate


def build_workload(tokenizer, args):
    if args.dataset_name == "random":
        return make_random_workload(
            tokenizer,
            num_prompts=args.num_prompts,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            seed=args.seed,
        )
    if args.dataset_name == "sharegpt":
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required for ShareGPT")
        return load_sharegpt_workload(
            tokenizer,
            args.dataset_path,
            num_prompts=args.num_prompts,
            seed=args.seed,
            max_model_len=args.max_model_len,
            max_output_len=args.max_output_len,
            fixed_output_len=args.sharegpt_output_len,
        )
    return make_prefix_repetition_workload(
        tokenizer,
        num_prompts=args.num_prompts,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
        output_len=args.prefix_output_len,
        seed=args.seed,
    )


def required_blocks(tokenizer, workload, block_size, concurrency):
    per_request = sorted(
        (
            math.ceil(
                (count_prompt_tokens(tokenizer, request.prompt) + request.output_tokens)
                / block_size
            )
            for request in workload.requests
        ),
        reverse=True,
    )
    return sum(per_request[:concurrency])


def _gib(num_bytes):
    return num_bytes / 1024**3


def print_report(report):
    result = report["result"]
    latency = result["latency_ms"]
    rate = report["benchmark"]["request_rate"]
    print("=" * 78)
    print(
        f"backend={report['engine']['model_backend']}  "
        f"dataset={report['workload']['name']}  requests={result['successful_requests']}  "
        f"request_rate={rate}  max_concurrency={result['max_concurrency_observed']}"
    )
    print("=" * 78)
    print(
        f"duration={result['benchmark_duration_s']:.3f}s  "
        f"requests/s={result['request_throughput_rps']:.3f}  "
        f"input tok/s={result['input_token_throughput_tps']:.1f}  "
        f"output tok/s={result['output_token_throughput_tps']:.1f}  "
        f"total tok/s={result['total_token_throughput_tps']:.1f}"
    )
    print(
        f"TTFT p50/p95/p99={latency['ttft'].get('p50', 0):.1f}/"
        f"{latency['ttft'].get('p95', 0):.1f}/{latency['ttft'].get('p99', 0):.1f}ms  "
        f"TPOT={latency['tpot'].get('p50', 0):.1f}/"
        f"{latency['tpot'].get('p95', 0):.1f}/{latency['tpot'].get('p99', 0):.1f}ms"
    )
    print(
        f"ITL p50/p95/p99={latency['itl'].get('p50', 0):.1f}/"
        f"{latency['itl'].get('p95', 0):.1f}/{latency['itl'].get('p99', 0):.1f}ms  "
        f"E2E={latency['e2e'].get('p50', 0):.1f}/"
        f"{latency['e2e'].get('p95', 0):.1f}/{latency['e2e'].get('p99', 0):.1f}ms"
    )
    print(
        f"cached prompt tokens={result['cached_prompt_tokens']}  "
        f"prefix hit fraction={result['prefix_cache_hit_fraction']:.3f}  "
        f"max pages used={result['max_physical_pages_used']}"
    )
    print(
        f"resident={_gib(result['resident_gpu_bytes']):.2f}GiB  "
        f"peak={_gib(result['peak_gpu_bytes']):.2f}GiB  "
        f"final blocks={result['final_block_state']}"
    )


def parser():
    values = argparse.ArgumentParser(
        description=(
            "Measure throughput, TTFT, TPOT, ITL, E2E, memory, pages, and "
            "prefix reuse using vLLM-style workload names."
        )
    )
    values.add_argument("model_path")
    values.add_argument(
        "--dataset-name",
        choices=["random", "sharegpt", "prefix_repetition"],
        default="random",
    )
    values.add_argument("--dataset-path")
    values.add_argument("--num-prompts", type=int, default=16)
    values.add_argument("--seed", type=int, default=17)
    values.add_argument("--request-rate", type=parse_request_rate, default=math.inf)
    values.add_argument("--max-concurrency", type=int, default=8)

    values.add_argument("--random-input-len", type=int, default=256)
    values.add_argument("--random-output-len", type=int, default=64)
    values.add_argument("--max-output-len", type=int, default=128)
    values.add_argument("--sharegpt-output-len", type=int)
    values.add_argument("--prefix-len", type=int, default=256)
    values.add_argument("--suffix-len", type=int, default=32)
    values.add_argument("--prefix-output-len", type=int, default=32)

    values.add_argument("--backend", choices=["paged", "hf"], default="paged")
    values.add_argument("--dtype", default="bfloat16")
    values.add_argument("--block-size", type=int, default=16)
    values.add_argument("--kv-cache-num-blocks", type=int, default=1024)
    values.add_argument("--max-model-len", type=int, default=4096)
    values.add_argument("--max-num-batched-tokens", type=int, default=2048)
    values.add_argument("--max-prefill-chunk-size", type=int, default=512)
    values.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    values.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    values.add_argument("--save-result")
    values.add_argument("--save-detailed", action="store_true")
    return values


def main():
    args = parser().parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    workload = build_workload(tokenizer, args)
    token_stats = workload_token_stats(tokenizer, workload)
    largest_context = max(
        count_prompt_tokens(tokenizer, request.prompt) + request.output_tokens
        for request in workload.requests
    )
    if largest_context > args.max_model_len:
        raise ValueError(
            f"workload needs max_model_len>={largest_context}, configured "
            f"{args.max_model_len}"
        )
    required = required_blocks(
        tokenizer,
        workload,
        args.block_size,
        min(args.max_concurrency, len(workload.requests)),
    )
    num_blocks = args.kv_cache_num_blocks or None
    if num_blocks is not None and num_blocks < required:
        raise ValueError(
            f"workload can require {required} active blocks; increase "
            f"--kv-cache-num-blocks from {num_blocks}"
        )
    prefix_caching = args.enable_prefix_caching
    if prefix_caching is None:
        prefix_caching = (
            args.backend == "paged" and args.dataset_name == "prefix_repetition"
        )
    if args.backend == "hf" and prefix_caching:
        raise ValueError("the HF reference path cannot enable physical prefix caching")

    engine = LLM(
        args.model_path,
        device="cuda",
        dtype=args.dtype,
        model_backend=args.backend,
        block_size=args.block_size,
        kv_cache_num_blocks=num_blocks,
        max_num_seqs=args.max_concurrency,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_prefix_caching=prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        decode_priority=True,
    )
    result = run_serving_workload(
        engine,
        workload,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        save_detailed=args.save_detailed,
    )
    report = {
        "schema_version": 1,
        "engine": {
            "model_path": str(args.model_path),
            "model_backend": args.backend,
            "dtype": args.dtype,
            "block_size": args.block_size,
            "kv_cache_num_blocks": engine.num_kv_blocks,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_concurrency,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "enable_prefix_caching": prefix_caching,
        },
        "benchmark": {
            "request_rate": (
                "inf" if math.isinf(args.request_rate) else args.request_rate
            ),
            "max_concurrency": args.max_concurrency,
            "seed": args.seed,
            "required_blocks_without_prefix_reuse": required,
        },
        "workload": {
            "name": workload.name,
            "metadata": workload.metadata,
            "token_stats": token_stats,
        },
        "result": result,
    }
    print_report(report)
    if args.save_result:
        output = Path(args.save_result)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved result: {output}")


if __name__ == "__main__":
    main()
