"""Diagnostic profiling: engine-level, request-level, and GPU kernel analysis.

Demonstrates the three-layer observability stack:
  Layer 1 (EngineProfiler) — per-step phase timing
  Layer 2 (RequestTracker) — TTFT, TPOT, E2E percentiles
  Layer 3 (PyTorch Profiler) — CUDA kernel attribution
  Diagnostic function — automatic bottleneck detection
"""

import sys

sys.path.insert(0, ".")

import torch
from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.observability.profiler import diagnose_slowdown


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"

    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "Python is a programming language",
    ]

    # Layer 1+2: Engine and request profiling
    print("=" * 60)
    print("  LAYER 1+2: Engine & Request Profiling")
    print("=" * 60)

    llm = LLM(model_path, enforce_eager=True)
    llm.enable_profiling()
    sp = SamplingParams(temperature=0.0, max_tokens=12)
    llm.generate(prompts, sp)

    timeline = llm._profiler.timeline()
    print("\n  Step timeline (first 3):")
    for t in timeline[:3]:
        print(
            f"    step {t['step']:3d} | sched: {t['schedule']:7.3f}ms | "
            f"prefill: {t['prefill']:7.3f}ms | decode: {t['decode']:7.3f}ms | "
            f"post: {t['post']:7.3f}ms | wq={t['waiting_q']} rq={t['running_q']}"
        )

    llm.profile_report()

    issues = diagnose_slowdown(llm._profiler, llm._request_tracker)
    print("  Diagnostic output:")
    for i, issue in enumerate(issues, 1):
        print(f"    [{i}] {issue}")
    print()

    # Layer 3: GPU kernel profiling
    print("=" * 60)
    print("  LAYER 3: GPU Kernel Profiling (PyTorch Profiler)")
    print("=" * 60)

    if torch.cuda.is_available():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            .cuda()
            .eval()
        )
        ids = tokenizer.encode(prompts[0], return_tensors="pt").cuda()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            with torch.no_grad():
                for _ in range(3):
                    _ = model(ids)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        print("  Skipped — no CUDA device available.\n")

    print("  Nsight Systems usage:")
    print("    nsys profile -o profile_out python script.py")
    print("    nsys-ui profile_out.nsys-rep")
    print("  Key Nsight signals:")
    print("    - Gaps between kernels → CPU/scheduler bound")
    print("    - Low SM occupancy → memory-bandwidth bound (decode)")
    print("    - High SM occupancy → compute bound (prefill/batch)")
    print("    - Many small kernel launches → need CUDA graphs")


if __name__ == "__main__":
    main()
