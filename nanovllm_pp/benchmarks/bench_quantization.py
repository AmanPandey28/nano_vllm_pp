"""Benchmark: educational weight-only INT8 storage and eager execution.

Compares the selected CUDA dtype with per-channel INT8 storage on the same
model. Both paths intentionally recompute full context. The INT8 module
dequantizes its full weight every forward, so this is not a fused INT8 kernel.
"""

import sys

import torch

sys.path.insert(0, ".")

from nanovllm_pp.device import (
    model_dtype_kwargs,
    resolve_device,
    resolve_dtype,
    timed_call,
)
from nanovllm_pp.quantization.int8_weight_only import (
    model_memory_report,
    quantize_model_int8,
)


def run_inference(model, tokenizer, prompt, max_tokens, device):
    def generate():
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            for _ in range(max_tokens):
                out = model(ids)
                logits = out.logits[:, -1, :]
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
        return ids

    generate()
    ids, elapsed = timed_call(generate, device=device)
    gen_text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return {
        "elapsed_s": round(elapsed, 3),
        "generated": gen_text[-200:],
        "token_ids": ids[0].tolist(),
    }


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
    device = resolve_device("cuda")
    dtype = resolve_dtype(device, "auto")
    dtype_label = {
        torch.bfloat16: "BF16",
        torch.float16: "FP16",
        torch.float32: "FP32",
    }[dtype]
    prompt = "The capital of France is"
    max_tokens = 8

    print(f"Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_dtype_kwargs(dtype),
        )
        .to(device)
        .eval()
    )

    baseline_mem = model_memory_report(model)
    baseline_result = run_inference(model, tokenizer, prompt, max_tokens, device)

    n_quant, n_skip = quantize_model_int8(model)
    int8_mem = model_memory_report(model)
    int8_result = run_inference(model, tokenizer, prompt, max_tokens, device)

    print("\n  Actual model storage (parameters + buffers):")
    print(f"    {dtype_label}: {baseline_mem['actual_mb']:.1f} MB")
    print(f"    INT8: {int8_mem['actual_mb']:.1f} MB")
    pct = 100 * int8_mem["actual_mb"] / max(baseline_mem["actual_mb"], 1e-9)
    print(f"    INT8 / {dtype_label}: {pct:.1f}%")
    print(f"    Layers quantized: {n_quant}, skipped: {n_skip}")
    print("\n  Inference:")
    print(f"    {dtype_label}: {baseline_result['elapsed_s']}s")
    print(f"    INT8: {int8_result['elapsed_s']}s")
    print(f"    {dtype_label} output: {baseline_result['generated']}")
    print(f"    INT8 output: {int8_result['generated']}")
    print(
        "    Exact token parity: "
        f"{'PASS' if baseline_result['token_ids'] == int8_result['token_ids'] else 'FAIL'}"
    )
