"""Benchmark: bitsandbytes NF4/INT8 inference vs BF16 baseline.

Compares:
  1. BF16 baseline
  2. NF4 quantization via bitsandbytes (double quant, 4-bit NormalFloat)
  3. INT8 quantization via bitsandbytes

Measures GPU memory usage, inference speed, and output quality.
"""

import statistics
import sys

import torch

sys.path.insert(0, ".")

from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.device import resolve_device, timed_call


def measure_memory(device):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / (1024**3)


def benchmark_generation(llm, prompts, max_tokens, repeats=3):
    warmup = SamplingParams(temperature=0.0, max_tokens=2)
    llm.generate([prompts[0]], warmup)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    samples = []
    outputs = None
    for _ in range(repeats):
        outputs, elapsed = timed_call(
            llm.generate, prompts, params, device=llm.device
        )
        samples.append(elapsed)
    return outputs, statistics.median(samples), samples


def run_bf16(model_path, prompts, max_tokens, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"\n  [BF16] Loading {model_path}...")
    llm = LLM(model_path, enforce_eager=True, device=str(device), dtype="bfloat16")
    gpu_used = measure_memory(device)
    outputs, elapsed, samples = benchmark_generation(llm, prompts, max_tokens)
    total_tokens = sum(o["num_completion_tokens"] for o in outputs)
    return {
        "method": "BF16",
        "gpu_gb": round(gpu_used, 2),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(total_tokens / elapsed, 1),
        "samples_s": [round(sample, 3) for sample in samples],
        "generated_token_ids": [
            output["token_ids"][output["num_prompt_tokens"] :] for output in outputs
        ],
        "output": outputs[0]["generated_text"][:100] if outputs else "",
    }


def run_nf4(model_path, prompts, max_tokens, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"\n  [NF4] Loading {model_path}...")
    llm = LLM(
        model_path,
        enforce_eager=True,
        device=str(device),
        dtype="bfloat16",
        load_in_4bit=True,
    )
    gpu_used = measure_memory(device)
    outputs, elapsed, samples = benchmark_generation(llm, prompts, max_tokens)
    total_tokens = sum(o["num_completion_tokens"] for o in outputs)
    return {
        "method": "NF4",
        "gpu_gb": round(gpu_used, 2),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(total_tokens / elapsed, 1),
        "samples_s": [round(sample, 3) for sample in samples],
        "generated_token_ids": [
            output["token_ids"][output["num_prompt_tokens"] :] for output in outputs
        ],
        "output": outputs[0]["generated_text"][:100] if outputs else "",
    }


def run_int8(model_path, prompts, max_tokens, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"\n  [INT8] Loading {model_path}...")
    llm = LLM(
        model_path,
        enforce_eager=True,
        device=str(device),
        dtype="bfloat16",
        load_in_8bit=True,
    )
    gpu_used = measure_memory(device)
    outputs, elapsed, samples = benchmark_generation(llm, prompts, max_tokens)
    total_tokens = sum(o["num_completion_tokens"] for o in outputs)
    return {
        "method": "INT8",
        "gpu_gb": round(gpu_used, 2),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(total_tokens / elapsed, 1),
        "samples_s": [round(sample, 3) for sample in samples],
        "generated_token_ids": [
            output["token_ids"][output["num_prompt_tokens"] :] for output in outputs
        ],
        "output": outputs[0]["generated_text"][:100] if outputs else "",
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if model_path is None:
        model_path = "/home/aman/dev/Projects/nano_vllm_++/Qwen3-0.6B"

    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "Python is a programming language",
    ]
    max_tokens = 12

    print(f"Model: {model_path}")
    print(f"Prompts: {len(prompts)}, max_tokens: {max_tokens}")
    print("=" * 60)
    device = resolve_device("cuda")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("this BF16/NF4 comparison requires BF16-capable CUDA")

    results = []
    results.append(run_bf16(model_path, prompts, max_tokens, device))
    results.append(run_nf4(model_path, prompts, max_tokens, device))

    try:
        results.append(run_int8(model_path, prompts, max_tokens, device))
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"\n  [INT8] Skipped: {e}")

    print(f"\n{'=' * 60}")
    print(
        f"  {'Method':<10} {'GPU (GB)':<12} {'Time (s)':<12} "
        f"{'tok/s':<10} {'Output (first 60 chars)'}"
    )
    print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 30}")
    for r in results:
        print(
            f"  {r['method']:<10} {r['gpu_gb']:<12} {r['elapsed_s']:<12} "
            f"{r['tok_per_s']:<10} {r['output'][:60]}"
        )
        print(f"    synchronized samples (s): {r['samples_s']}")
        exact_matches = sum(
            candidate == reference
            for candidate, reference in zip(
                r["generated_token_ids"], results[0]["generated_token_ids"]
            )
        )
        print(f"    exact BF16 token matches: {exact_matches}/{len(prompts)}")

    if len(results) >= 2:
        mem_saved = results[0]["gpu_gb"] - results[1]["gpu_gb"]
        pct = mem_saved / results[0]["gpu_gb"] * 100
        print(f"\n  NF4 saves {mem_saved:.1f} GB ({pct:.0f}%) vs BF16")

    print("\n  bitsandbytes NF4 info:")
    print("    - 4-bit NormalFloat quantization (information-theoretically optimal)")
    print("    - Double quantization: quantizes the scale factors too")
    print("    - Blocksize 64: each block has its own scale")
    print("    - Compute dtype: bfloat16 (dequantized on the fly)")
