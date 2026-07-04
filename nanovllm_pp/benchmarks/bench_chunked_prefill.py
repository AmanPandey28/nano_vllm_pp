"""Benchmark: chunked prefill with mixed long + short prompt workloads.

Compares enable_chunked_prefill=True vs False on the same batch to
measure TTFT improvement when long prompts are split into chunks.
"""

import time
import sys

sys.path.insert(0, ".")

from nanovllm_pp import LLM, SamplingParams


def run_benchmark(model_path, enable_chunked, label, prompts, max_tokens=16):
    print(f"\n  [{label}] enable_chunked_prefill={enable_chunked}")
    llm = LLM(
        model_path,
        enforce_eager=True,
        enable_chunked_prefill=enable_chunked,
        max_prefill_chunk_size=512,
        max_num_batched_tokens=2048,
        decode_priority=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    start = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - start
    total_gen = sum(o["num_completion_tokens"] for o in outputs)
    print(
        f"    Time: {elapsed:.2f}s, Steps: {llm.step_counter}, "
        f"Gen tokens: {total_gen}, tok/s: {total_gen / elapsed:.1f}"
    )
    return {
        "label": label,
        "elapsed_s": round(elapsed, 3),
        "engine_steps": llm.step_counter,
        "total_gen_tokens": total_gen,
        "tok_per_sec": round(total_gen / elapsed, 1),
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

    no_chunk = run_benchmark(model_path, False, "No chunking", prompts, max_tokens)
    with_chunk = run_benchmark(model_path, True, "Chunked (512)", prompts, max_tokens)

    print(f"\n{'=' * 60}")
    print(f"  {'Method':<20} {'Time(s)':<10} {'Steps':<10} {'tok/s':<10}")
    for r in [no_chunk, with_chunk]:
        print(
            f"  {r['label']:<20} {r['elapsed_s']:<10} "
            f"{r['engine_steps']:<10} {r['tok_per_sec']:<10}"
        )
