"""Tutorial: full NanoVLLM++ engine pipeline.

Demonstrates prefill/decode, iteration-level scheduling, sequence lifecycle,
and logical block accounting end-to-end. The HF cache is not physically paged.
"""

import sys
import time

sys.path.insert(0, ".")

from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.device import resolve_device, synchronize

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"

    print(f"Loading {model_path}...")
    device = resolve_device("cuda")
    llm = LLM(model_path, enforce_eager=True, device=str(device))

    prompts = [
        "The capital of France is",
        "The largest planet is",
        "Python is a language for",
    ]
    sp = SamplingParams(temperature=0.7, max_tokens=24)

    print(f"Generating for {len(prompts)} prompts...")
    synchronize(device)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    synchronize(device)
    elapsed = time.perf_counter() - start

    for i, out in enumerate(outputs):
        print(f"\n--- Prompt {i} ---")
        print(f"  Request: {out['request_id']}")
        print(f"  Prompt tokens: {out['num_prompt_tokens']}")
        print(f"  Completion tokens: {out['num_completion_tokens']}")
        print(f"  Generated: {out['generated_text'][:200]}")

    print(f"\nTotal: {elapsed:.2f}s, Steps: {llm.step_counter}")
    s = llm.scheduler.debug_state()
    print(f"Scheduler: waiting={s['waiting']}, running={s['running']}")
    print(f"Blocks: {s['blocks']}")
