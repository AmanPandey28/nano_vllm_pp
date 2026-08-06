"""End-to-end correctness and physical-prefix demo for the paged backend."""

import argparse
import gc

import torch

from nanovllm_pp import LLM, SamplingParams

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Paged attention maps logical blocks to",
]


def generated_ids(outputs):
    return [output["token_ids"][output["num_prompt_tokens"] :] for output in outputs]


def build_engine(model_path, backend, args, **overrides):
    options = {
        "device": "cuda",
        "dtype": "bfloat16",
        "model_backend": backend,
        "block_size": args.block_size,
        "kv_cache_num_blocks": args.kv_cache_num_blocks,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 128,
        "max_model_len": 512,
    }
    options.update(overrides)
    return LLM(model_path, **options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--kv-cache-num-blocks", type=int, default=64)
    parser.add_argument("--skip-hf-parity", action="store_true")
    args = parser.parse_args()
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)

    reference_ids = None
    if not args.skip_hf_parity:
        reference = build_engine(args.model_path, "hf", args)
        reference_ids = generated_ids(reference.generate(DEFAULT_PROMPTS, params))
        del reference
        gc.collect()
        torch.cuda.empty_cache()

    paged = build_engine(
        args.model_path,
        "paged",
        args,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_prefill_chunk_size=16,
    )
    outputs = paged.generate(DEFAULT_PROMPTS, params)
    paged_ids = generated_ids(outputs)
    print(f"physical KV shape: {tuple(paged.model_runner.kv_cache.shape)}")
    print(
        "physical KV bytes: "
        f"{paged.model_runner.kv_cache.numel() * paged.model_runner.kv_cache.element_size():,}"
    )
    for prompt, output in zip(DEFAULT_PROMPTS, outputs):
        print(f"{prompt!r} -> {output['generated_text']!r}")
    if reference_ids is not None:
        print(f"HF greedy token parity: {paged_ids == reference_ids}")

    first = (
        "Qualcomm accelerator optimization requires careful memory layout "
        "and kernel fusion. First"
    )
    second = (
        "Qualcomm accelerator optimization requires careful memory layout "
        "and kernel fusion. Second"
    )
    short = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
    paged.generate(first, short)
    request_id = paged.add_request(second, short)
    paged.step()
    sequence = paged.get_request(request_id)
    print(
        "physical prefix reuse: "
        f"cached_tokens={sequence.num_cached_tokens}, "
        f"computed_tokens={sequence.num_computed_tokens}, "
        f"block_table={sequence.block_table}"
    )
    while paged.scheduler.has_unfinished():
        paged.step()
    print(f"final block state: {paged.block_manager.stats()}")


if __name__ == "__main__":
    main()
