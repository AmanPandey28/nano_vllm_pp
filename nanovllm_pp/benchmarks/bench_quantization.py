"""Benchmark: weight-only INT8 quantization — memory and speed.

Compares FP16 baseline with per-channel INT8 on the same model.
Measures weight memory reduction and inference time.
"""

import time
import sys
import torch

sys.path.insert(0, ".")

from nanovllm_pp.quantization.int8_weight_only import (
    quantize_model_int8,
    model_memory_report,
)


def run_inference(model, tokenizer, prompt, max_tokens, device):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    start = time.time()
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    elapsed = time.time() - start
    gen_text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return {"elapsed_s": round(elapsed, 3), "generated": gen_text[-200:]}


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    prompt = "The capital of France is"
    max_tokens = 8

    print(f"Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )

    fp16_mem = model_memory_report(model)
    fp16_result = run_inference(model, tokenizer, prompt, max_tokens, device)

    n_quant, n_skip = quantize_model_int8(model)
    int8_mem = model_memory_report(model)
    int8_result = run_inference(model, tokenizer, prompt, max_tokens, device)

    print(f"\n  Memory:")
    print(f"    FP16 params: {fp16_mem['fp16_mb']:.1f} MB")
    print(
        f"    INT8 params: {int8_mem['int8_mb']:.1f} MB "
        f"({int8_mem['int8_vs_fp16']} of FP16)"
    )
    print(
        f"    INT4 estimate: {int8_mem['int4_mb']:.1f} MB "
        f"({int8_mem['int4_vs_fp16']} of FP16)"
    )
    print(f"    Layers quantized: {n_quant}, skipped: {n_skip}")
    print(f"\n  Inference:")
    print(f"    FP16: {fp16_result['elapsed_s']}s")
    print(f"    INT8: {int8_result['elapsed_s']}s")
    print(f"    FP16 output: {fp16_result['generated']}")
    print(f"    INT8 output: {int8_result['generated']}")
