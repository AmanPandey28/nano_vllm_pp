"""Small paged-backend example. Run from an editable install, on CUDA."""

import argparse

from nanovllm_pp import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    args = parser.parse_args()
    llm = LLM(
        args.model_path, device="cuda", dtype="bfloat16", model_backend="paged",
        block_size=16, kv_cache_num_blocks=256, max_num_seqs=4,
        max_num_batched_tokens=256, max_model_len=1024,
        enable_chunked_prefill=True, max_prefill_chunk_size=128,
        enable_prefix_caching=True,
    )
    outputs = llm.generate(
        ["The capital of France is", "A transformer KV cache stores"],
        SamplingParams(temperature=0, max_tokens=32),
    )
    for output in outputs:
        print(f"Output: {output['generated_text']}")
        print(f"Tokens: {output['num_completion_tokens']}; stop: {output['finish_reason']}")


if __name__ == "__main__":
    main()
