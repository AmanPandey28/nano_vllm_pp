"""Shared implementation for the one-command terminal benchmark demos.

The public entry points in this directory intentionally keep their configuration
small and explicit.  They all run the same deterministic, shared-prefix workload
so their terminal reports can be compared without changing benchmark shape.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers.utils import logging as transformers_logging

from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.benchmarks.serving_harness import run_serving_workload
from nanovllm_pp.benchmarks.workloads import (
    BenchmarkRequest,
    BenchmarkWorkload,
    common_prefix_tokens,
    count_prompt_tokens,
    workload_token_stats,
)

# Some third-party transformer integrations still call a PyTorch API that emits
# a deprecation log during model loading. It is unrelated to eager inference and
# would otherwise obscure the benchmark header.
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
transformers_logging.disable_progress_bar()


@dataclass(frozen=True)
class DemoVariant:
    """The backend and scheduler features isolated by one demo script."""

    key: str
    title: str
    backend: str
    chunked_prefill: bool
    prefix_caching: bool


PAGED_BASELINE = DemoVariant(
    key="paged-baseline",
    title="Paged backend: chunked prefill OFF, prefix caching OFF",
    backend="paged",
    chunked_prefill=False,
    prefix_caching=False,
)

PAGED_OPTIMIZED = DemoVariant(
    key="paged-optimized",
    title="Paged backend: chunked prefill ON, prefix caching ON",
    backend="paged",
    chunked_prefill=True,
    prefix_caching=True,
)

HF_REFERENCE = DemoVariant(
    key="hf-reference",
    title="Hugging Face backend: chunked prefill ON, prefix caching OFF",
    backend="hf",
    chunked_prefill=True,
    prefix_caching=False,
)


_SHARED_PREFIX = """
You are explaining a compact language-model inference engine to a systems
engineer. The engine separates prompt prefill from autoregressive decode. Its
continuous-batching scheduler selects work under sequence-count, token, and KV
memory budgets. A block manager maps logical token positions to fixed-size
physical KV-cache pages. The paged model runner builds slot mappings and block
tables, writes keys and values into a global cache, runs attention, and samples
the next token. Give a direct, technically precise answer in one sentence.
""".strip()

_QUESTIONS = (
    "What responsibility belongs to the scheduler, and what belongs to the block manager?",
    "Why does paged KV caching use both a block table and a slot mapping?",
    "How can chunked prefill improve decode responsiveness under mixed load?",
    "Why can higher batching throughput increase time to first token?",
)


def _default_model() -> str:
    """Prefer an explicit model, then the known local checkout, then HF Hub."""
    configured = os.environ.get("NVPP_MODEL")
    if configured:
        return configured

    repository = Path(__file__).resolve().parents[2]
    candidates = (
        repository.parent / "nano_vllm_++" / "Qwen3-0.6B",
        repository / "Qwen3-0.6B",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return "Qwen/Qwen3-0.6B"


def build_workload(tokenizer, *, num_prompts: int, output_tokens: int) -> BenchmarkWorkload:
    """Create a readable workload whose requests share an exact prompt prefix."""
    requests = []
    for index in range(num_prompts):
        question = _QUESTIONS[index % len(_QUESTIONS)]
        if index >= len(_QUESTIONS):
            question = f"{question} Discuss request variant {index + 1}."
        requests.append(
            BenchmarkRequest(
                prompt=f"{_SHARED_PREFIX}\n\nQuestion: {question}\nAnswer:",
                output_tokens=output_tokens,
                source=question,
            )
        )

    warmup_question = (
        "What roles do the scheduler, block manager, and model runner play?"
    )
    warmup = BenchmarkRequest(
        prompt=(
            f"{_SHARED_PREFIX}\n\n"
            f"Question: {warmup_question}\nAnswer:"
        ),
        output_tokens=output_tokens,
        source=warmup_question,
    )
    prompts = [warmup.prompt, *(request.prompt for request in requests)]
    return BenchmarkWorkload(
        name="live-shared-prefix",
        requests=tuple(requests),
        warmup_requests=(warmup,),
        metadata={
            "actual_common_prefix_tokens": common_prefix_tokens(tokenizer, prompts),
            "output_tokens_per_request": output_tokens,
        },
    )


def _gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


def _latency_row(name: str, values: dict) -> str:
    return (
        f"{name:<8}"
        f"{values.get('mean', 0.0):>12.1f}"
        f"{values.get('p50', 0.0):>12.1f}"
        f"{values.get('p95', 0.0):>12.1f}"
        f"{values.get('p99', 0.0):>12.1f}"
    )


def _print_outputs(engine, workload: BenchmarkWorkload, result: dict) -> None:
    print("\nMEASURED OUTPUT SAMPLES")
    print("-" * 78)
    print(
        "These samples use a fixed output-token count for comparable work; "
        "a final sentence may therefore be truncated.\n"
    )
    for index, (request, record) in enumerate(
        zip(workload.requests, result["requests"]), start=1
    ):
        generated = engine.tokenizer.decode(
            record["generated_token_ids"], skip_special_tokens=True
        ).strip()
        print(f"[{index}] Prompt: {request.source}")
        print(f"    Output: {generated or '<only special tokens were generated>'}\n")


def _print_report(
    variant: DemoVariant,
    result: dict,
) -> None:
    latency = result["latency_ms"]
    profile = result["engine_profile_ms"]
    blocks = result["final_block_state"]
    cuda = result["cuda"]

    print("\n" + "=" * 78)
    print("MEASURED RESULTS (warm-up excluded)")
    print("=" * 78)
    print(
        f"GPU: {cuda['name']}  |  Compute capability: {cuda['compute_capability']}  |  "
        f"PyTorch/CUDA: {cuda['torch_version']}/{cuda['torch_cuda_version']}"
    )
    print(
        f"Requests: {result['successful_requests']}  |  "
        f"Duration: {result['benchmark_duration_s']:.3f} s  |  "
        f"Maximum concurrency: {result['max_concurrency_observed']}"
    )
    print(
        f"Request throughput: {result['request_throughput_rps']:.2f} req/s\n"
        f"Input throughput:   {result['input_token_throughput_tps']:.1f} tok/s\n"
        f"Output throughput:  {result['output_token_throughput_tps']:.1f} tok/s\n"
        f"Total throughput:   {result['total_token_throughput_tps']:.1f} tok/s"
    )

    print("\nLATENCY (milliseconds)")
    print(f"{'Metric':<8}{'Mean':>12}{'p50':>12}{'p95':>12}{'p99':>12}")
    print("-" * 56)
    for metric in ("ttft", "tpot", "itl", "e2e"):
        print(_latency_row(metric.upper(), latency[metric]))

    cached = result["cached_prompt_tokens"]
    total_input = result["total_input_tokens"]
    print("\nKV CACHE AND PREFIX REUSE")
    print(
        f"Cached prompt tokens: {cached}/{total_input}  |  "
        f"Hit fraction: {result['prefix_cache_hit_fraction']:.3f}"
    )
    page_label = "physical" if variant.backend == "paged" else "logical admission"
    print(
        f"Maximum {page_label} pages reserved: "
        f"{result['max_physical_pages_used']}"
    )
    print(
        f"Final block state: total={blocks['total_blocks']}, "
        f"used={blocks['used_blocks']}, free={blocks['free_blocks']}, "
        f"cached_hashes={blocks['cached_hashes']}"
    )
    if "physical_kv_cache" in result:
        cache = result["physical_kv_cache"]
        print(
            f"Physical KV tensor: shape={cache['shape']}  |  "
            f"size={_gib(cache['bytes']):.2f} GiB"
        )
    else:
        print("Physical KV tensor: per-request Hugging Face DynamicCache")

    print("\nGPU MEMORY")
    print(
        f"Resident after warm-up: {_gib(result['resident_gpu_bytes']):.2f} GiB  |  "
        f"Measured peak: {_gib(result['peak_gpu_bytes']):.2f} GiB"
    )

    print("\nENGINE PHASE PROFILE (average per engine step)")
    print(
        f"schedule={profile.get('avg_schedule_ms', 0):.3f} ms  |  "
        f"prefill={profile.get('avg_prefill_ms', 0):.3f} ms  |  "
        f"decode={profile.get('avg_decode_ms', 0):.3f} ms  |  "
        f"mixed={profile.get('avg_mixed_ms', 0):.3f} ms  |  "
        f"postprocess={profile.get('avg_postprocess_ms', 0):.3f} ms"
    )
    print(
        f"Engine steps: {profile.get('steps', 0)}  |  "
        f"Largest measured phase: {profile.get('bottleneck', 'n/a')}"
    )


def _build_parser(variant: DemoVariant) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the {variant.key} terminal generation benchmark."
    )
    parser.add_argument(
        "--model",
        default=_default_model(),
        help=(
            "Qwen3-0.6B checkpoint path or Hugging Face ID. Defaults to "
            "$NVPP_MODEL, a detected sibling checkout, or Qwen/Qwen3-0.6B."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-cache-num-blocks", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-prefill-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Optionally save the complete machine-readable report.",
    )
    return parser


def _validate_args(args) -> None:
    positive = {
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "max_concurrency": args.max_concurrency,
        "block_size": args.block_size,
        "kv_cache_num_blocks": args.kv_cache_num_blocks,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_prefill_chunk_size": args.max_prefill_chunk_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these options must be positive: {', '.join(invalid)}")


def run_demo(variant: DemoVariant) -> None:
    args = _build_parser(variant).parse_args()
    _validate_args(args)

    print("=" * 78)
    print(variant.title)
    print("=" * 78)
    print(f"Model: {args.model}")
    print("Loading model and allocating cache; the first request is a warm-up...")

    engine = LLM(
        args.model,
        device="cuda",
        dtype=args.dtype,
        model_backend=variant.backend,
        block_size=args.block_size,
        kv_cache_num_blocks=args.kv_cache_num_blocks,
        max_num_seqs=args.max_concurrency,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=variant.chunked_prefill,
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        enable_prefix_caching=variant.prefix_caching,
        decode_priority=True,
        enforce_eager=True,
    )
    workload = build_workload(
        engine.tokenizer,
        num_prompts=args.num_prompts,
        output_tokens=args.max_tokens,
    )
    token_stats = workload_token_stats(engine.tokenizer, workload)
    largest_context = max(
        count_prompt_tokens(engine.tokenizer, request.prompt) + request.output_tokens
        for request in workload.requests
    )
    if largest_context > args.max_model_len:
        raise ValueError(
            f"workload requires max_model_len >= {largest_context}; configured "
            f"value is {args.max_model_len}"
        )
    per_request_blocks = sorted(
        (
            math.ceil(
                (
                    count_prompt_tokens(engine.tokenizer, request.prompt)
                    + request.output_tokens
                )
                / args.block_size
            )
            for request in workload.requests
        ),
        reverse=True,
    )
    required_blocks = sum(per_request_blocks[: args.max_concurrency])
    if required_blocks > args.kv_cache_num_blocks:
        raise ValueError(
            "workload may need "
            f"{required_blocks} blocks without prefix sharing, but only "
            f"{args.kv_cache_num_blocks} are configured"
        )

    print(f"Device: {engine.device}  |  Compute dtype: {engine.compute_dtype}")
    print(
        f"Backend: {variant.backend}  |  "
        f"Chunked prefill: {variant.chunked_prefill}  |  "
        f"Prefix caching: {variant.prefix_caching}"
    )
    print(
        f"Workload: {args.num_prompts} requests x {args.max_tokens} output tokens  |  "
        f"Input tokens min/mean/max: "
        f"{token_stats['input_tokens']['min']}/"
        f"{token_stats['input_tokens']['mean']}/"
        f"{token_stats['input_tokens']['max']}"
    )
    print(
        f"Shared prefix: {workload.metadata['actual_common_prefix_tokens']} tokens  |  "
        f"Concurrency: {args.max_concurrency}  |  "
        f"Token budget: {args.max_num_batched_tokens}/step"
    )
    print(
        f"Block size/cache blocks: {args.block_size}/{args.kv_cache_num_blocks}  |  "
        f"Maximum prefill chunk: {args.max_prefill_chunk_size}"
    )
    if variant.backend == "hf":
        print(
            "Note: HF block counts are scheduler admission metadata; actual KV "
            "storage is a per-request DynamicCache."
        )

    showcase_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        ignore_eos=False,
        stop="\n",
    )
    showcase_request = workload.warmup_requests[0]
    showcase = engine.generate(showcase_request.prompt, showcase_params)[0]
    print("\nUNTIMED GENERATION CHECK")
    print("-" * 78)
    print(f"Prompt: {showcase_request.source}")
    print(f"Output: {showcase['generated_text'].strip()}")
    print(f"Finish reason: {showcase['finish_reason']}")

    result = run_serving_workload(
        engine,
        workload,
        request_rate=math.inf,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        save_detailed=True,
    )
    _print_outputs(engine, workload, result)
    _print_report(variant, result)

    report = {
        "schema_version": 1,
        "variant": asdict(variant),
        "engine": {
            "model": str(args.model),
            "device": engine.device,
            "compute_dtype": str(engine.compute_dtype),
            "block_size": args.block_size,
            "kv_cache_num_blocks": engine.num_kv_blocks,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_concurrency,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_prefill_chunk_size": args.max_prefill_chunk_size,
        },
        "workload": {
            "name": workload.name,
            "metadata": workload.metadata,
            "token_stats": token_stats,
        },
        "result": result,
    }
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON report: {args.save_json}")

    print("\nInterpretation note: compare runs only when model, workload, power mode,")
    print("software versions, and all numeric flags are identical.")
