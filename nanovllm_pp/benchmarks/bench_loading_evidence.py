"""Quantized-loading storage, allocated memory, latency, and fidelity evidence.

All formats use the HF runner. Custom INT8/INT4 modules dequantize to BF16
before matmul; these are storage prototypes, not fused integer compute.
"""

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.device import synchronize, timed_call
from nanovllm_pp.quantization.int8_weight_only import (
    model_memory_report,
    quantize_model_int4,
    quantize_model_int8,
)

from .evidence import environment_manifest

PROMPTS = (
    "The capital of France is",
    "Machine learning is a field of",
    "Python is a programming language",
)
QUALITY_TEXTS = (
    (
        "A transformer predicts the next token from the tokens it has already seen. "
        "The key and value cache avoids recomputing representations of earlier tokens."
    ),
    (
        "An inference scheduler balances latency, throughput, and memory capacity. "
        "Requests may have different prompt lengths and generate different numbers of tokens."
    ),
    (
        "Quantization reduces the storage needed for neural network weights. "
        "Numerical error and extra conversion work can affect accuracy and execution time."
    ),
)


@torch.inference_mode()
def quality_probe(engine):
    last_logits = []
    total_nll = 0.0
    total_tokens = 0
    finite = True
    for text in QUALITY_TEXTS:
        ids = torch.tensor([engine.tokenizer.encode(text)], device=engine.device)
        logits = engine.model(input_ids=ids, use_cache=False).logits.float()
        finite = finite and bool(torch.isfinite(logits).all())
        total_nll += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        total_tokens += ids.shape[-1] - 1
        last_logits.append(logits[0, -1].cpu())
    return last_logits, {
        "finite_logits": finite,
        "teacher_forced_tokens": total_tokens,
        "mean_token_nll": total_nll / total_tokens,
        "scope": "three fixed synthetic texts; diagnostic only, not a task accuracy or corpus perplexity score",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument(
        "--output", default="runs/quantized_loading.json"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats and max-tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be visible for loading measurements")
    manifest = environment_manifest(args.model_path)
    report = {
        "manifest": manifest,
        "prompts": PROMPTS,
        "quality_texts": QUALITY_TEXTS,
        "output_tokens_per_request": args.max_tokens,
        "formats": {},
    }
    reference_logits = None
    reference_ids = None
    for method in ("bf16", "nf4", "bnb_int8", "int8_storage", "int4_storage"):
        gc.collect()
        torch.cuda.empty_cache()
        synchronize("cuda")
        background = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        kwargs = {"load_in_4bit": method == "nf4", "load_in_8bit": method == "bnb_int8"}
        engine, load_seconds = timed_call(
            LLM,
            args.model_path,
            device="cuda",
            dtype="bfloat16",
            model_backend="hf",
            kv_cache_num_blocks=128,
            block_size=16,
            max_model_len=512,
            **kwargs,
        )
        converted = None
        if method in {"int8_storage", "int4_storage"}:
            convert = (
                quantize_model_int8 if method == "int8_storage" else quantize_model_int4
            )
            converted, conversion_seconds = timed_call(
                convert, engine.model, device="cuda"
            )
            load_seconds += conversion_seconds
        synchronize("cuda")
        loaded_bytes = torch.cuda.memory_allocated() - background
        load_peak_bytes = torch.cuda.max_memory_allocated() - background
        storage = model_memory_report(engine.model)
        params = SamplingParams(
            temperature=0, max_tokens=args.max_tokens, ignore_eos=True
        )
        engine.generate(PROMPTS, params)
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.repeats):
            outputs, elapsed = timed_call(
                engine.generate, PROMPTS, params, device="cuda"
            )
            samples.append(elapsed)
        generation_peak = torch.cuda.max_memory_allocated() - background
        ids = [out["token_ids"][out["num_prompt_tokens"] :] for out in outputs]
        logits, quality = quality_probe(engine)
        if reference_logits is None:
            reference_logits, reference_ids = logits, ids
        quality.update(
            {
                "last_logit_cosine_vs_bf16": [
                    F.cosine_similarity(a, b, dim=0).item()
                    for a, b in zip(logits, reference_logits)
                ],
                "last_logit_max_abs_error_vs_bf16": [
                    float((a - b).abs().max()) for a, b in zip(logits, reference_logits)
                ],
                "exact_greedy_completion_matches_vs_bf16": sum(
                    a == b for a, b in zip(ids, reference_ids)
                ),
            }
        )
        entry = {
            "loaded_torch_allocated_bytes": loaded_bytes,
            "loaded_torch_allocated_gib": loaded_bytes / 1024**3,
            "peak_load_torch_allocated_bytes": load_peak_bytes,
            "peak_generation_torch_allocated_bytes": generation_peak,
            "counted_parameter_buffer_storage": storage,
            "load_and_conversion_seconds": load_seconds,
            "generation_samples_seconds": samples,
            "median_generation_seconds": statistics.median(samples),
            "output_tokens_per_second": len(PROMPTS)
            * args.max_tokens
            / statistics.median(samples),
            "quality": quality,
            "converted_skipped_linear_counts": converted,
            "generated_token_ids": ids,
            "generated_text": [out["generated_text"] for out in outputs],
        }
        report["formats"][method] = entry
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"{method}: loaded={entry['loaded_torch_allocated_gib']:.3f} GiB; "
            f"rate={entry['output_tokens_per_second']:.1f} output tok/s; "
            f"BF16 matches={quality['exact_greedy_completion_matches_vs_bf16']}/{len(PROMPTS)}; "
            f"NLL={quality['mean_token_nll']:.4f}",
            flush=True,
        )
        del engine, logits, outputs
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
