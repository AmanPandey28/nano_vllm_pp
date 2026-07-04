"""Benchmark: NF4/FP4 quantized inference vs FP16 baseline.

Compares:
  1. FP16 baseline
  2. NF4 quantization via bitsandbytes (double quant, 4-bit NormalFloat)
  3. INT8 quantization via bitsandbytes

Measures GPU memory usage, inference speed, and output quality.
"""

import sys
import time
import torch

sys.path.insert(0, ".")

from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.quantization.int8_weight_only import model_memory_report


def measure_memory():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024**3)


def run_fp16(model_path, prompts, max_tokens):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"\n  [FP16] Loading {model_path}...")
    llm = LLM(model_path, enforce_eager=True)
    gpu_used = measure_memory()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    start = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - start
    total_tokens = sum(o["num_completion_tokens"] for o in outputs)
    return {
        "method": "FP16",
        "gpu_gb": round(gpu_used, 2),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(total_tokens / elapsed, 1),
        "output": outputs[0]["generated_text"][:100] if outputs else "",
    }


def run_nf4(model_path, prompts, max_tokens):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"\n  [NF4] Loading {model_path}...")
    llm = LLM(model_path, enforce_eager=True, load_in_4bit=True)
    gpu_used = measure_memory()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    start = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - start
    total_tokens = sum(o["num_completion_tokens"] for o in outputs)
    return {
        "method": "NF4",
        "gpu_gb": round(gpu_used, 2),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(total_tokens / elapsed, 1),
        "output": outputs[0]["generated_text"][:100] if outputs else "",
    }


def run_int8(model_path, prompts, max_tokens):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"\n  [INT8] Loading {model_path}...")
    llm = LLM(model_path, enforce_eager=True, load_in_8bit=True)
    gpu_used = measure_memory()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    start = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - start
    total_tokens = sum(o["num_completion_tokens"] for o in outputs)
    return {
        "method": "INT8",
        "gpu_gb": round(gpu_used, 2),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(total_tokens / elapsed, 1),
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

    results = []
    results.append(run_fp16(model_path, prompts, max_tokens))
    results.append(run_nf4(model_path, prompts, max_tokens))

    try:
        results.append(run_int8(model_path, prompts, max_tokens))
    except Exception as e:
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

    if len(results) >= 2:
        mem_saved = results[0]["gpu_gb"] - results[1]["gpu_gb"]
        pct = mem_saved / results[0]["gpu_gb"] * 100
        print(f"\n  NF4 saves {mem_saved:.1f} GB ({pct:.0f}%) vs FP16")

    print(f"\n  bitsandbytes NF4 info:")
    print(f"    - 4-bit NormalFloat quantization (information-theoretically optimal)")
    print(f"    - Double quantization: quantizes the scale factors too")
    print(f"    - Blocksize 64: each block has its own scale")
    print(f"    - Compute dtype: bfloat16 (dequantized on the fly)")
