"""Benchmark: chunked prefill with mixed long + short prompt workloads.

Compares enable_chunked_prefill=True vs False on the same batch to
measure TTFT improvement when long prompts are split into chunks.
"""

import sys
import time

sys.path.insert(0, ".")

from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.device import resolve_device, synchronize


def run_benchmark(
    model_path, enable_chunked, label, prompts, max_tokens=16, chunk_size=64
):
    print(f"\n  [{label}] enable_chunked_prefill={enable_chunked}")
    device = resolve_device("cuda")
    llm = LLM(
        model_path,
        device=str(device),
        enforce_eager=True,
        enable_chunked_prefill=enable_chunked,
        max_prefill_chunk_size=chunk_size,
        max_num_batched_tokens=2048,
        decode_priority=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    llm.generate(["CUDA warm-up"], SamplingParams(temperature=0.0, max_tokens=2))
    llm.enable_profiling()
    steps_before = llm.step_counter
    synchronize(device)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    synchronize(device)
    elapsed = time.perf_counter() - start
    total_gen = sum(o["num_completion_tokens"] for o in outputs)
    request_timings = list(llm._request_tracker.requests.values())
    ttft_ms = [
        (request["first_token_t"] - request["enqueue_t"]) * 1000
        for request in request_timings
    ]
    long_ttft_ms = ttft_ms[0]
    short_ttft_mean_ms = sum(ttft_ms[1:]) / len(ttft_ms[1:])
    print(
        f"    Time: {elapsed:.2f}s, Steps: {llm.step_counter - steps_before}, "
        f"Gen tokens: {total_gen}, tok/s: {total_gen / elapsed:.1f}\n"
        f"    TTFT long: {long_ttft_ms:.1f}ms, "
        f"short mean: {short_ttft_mean_ms:.1f}ms"
    )
    return {
        "label": label,
        "elapsed_s": round(elapsed, 3),
        "engine_steps": llm.step_counter - steps_before,
        "total_gen_tokens": total_gen,
        "tok_per_sec": round(total_gen / elapsed, 1),
        "long_ttft_ms": round(long_ttft_ms, 1),
        "short_ttft_mean_ms": round(short_ttft_mean_ms, 1),
        "generated_token_ids": [
            output["token_ids"][output["num_prompt_tokens"] :] for output in outputs
        ],
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"

    long_prompt = (
        "The history of artificial intelligence dates back to ancient "
        "Greek myths of mechanical servants. In the modern era, Alan "
        "Turing's 1950 paper proposed the Turing test. The field was "
        "formally founded at the Dartmouth Conference in 1956 by John "
        "McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude "
        "Shannon. The early decades saw optimism followed by the first "
        "AI winter. The 1980s brought expert systems, but a second AI "
        "winter followed. The 2010s marked a renaissance driven by deep "
        "learning, GPU computing, and large datasets."
    )

    prompts = [long_prompt] + [
        "The capital of Japan is",
        "Python was created by",
        "Machine learning is",
        "The speed of light is",
        "Water boils at",
    ]
    max_tokens = 12

    print(f"Model: {model_path}")
    print(
        f"Workload: 1 long prompt + {len(prompts) - 1} short prompts, "
        f"max_tokens={max_tokens}"
    )
    print("=" * 60)

    chunk_size = 64
    no_chunk = run_benchmark(
        model_path, False, "No chunking", prompts, max_tokens, chunk_size
    )
    with_chunk = run_benchmark(
        model_path, True, f"Chunked ({chunk_size})", prompts, max_tokens, chunk_size
    )
    if no_chunk["generated_token_ids"] != with_chunk["generated_token_ids"]:
        raise RuntimeError("chunked prefill changed greedy output tokens")

    print(f"\n{'=' * 60}")
    print(
        f"  {'Method':<20} {'Time(s)':<10} {'Steps':<8} {'tok/s':<9} "
        f"{'Long TTFT':<11} {'Short TTFT'}"
    )
    for r in [no_chunk, with_chunk]:
        print(
            f"  {r['label']:<20} {r['elapsed_s']:<10} "
            f"{r['engine_steps']:<8} {r['tok_per_sec']:<9} "
            f"{r['long_ttft_ms']:<11} {r['short_ttft_mean_ms']}"
        )
    print("\n  Chunked/non-chunked greedy token parity: PASS")
