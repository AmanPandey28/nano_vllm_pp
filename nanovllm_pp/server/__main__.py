"""Launch one model instance: python -m nanovllm_pp.server MODEL --help."""

import argparse


def parser():
    values = argparse.ArgumentParser(description=__doc__)
    values.add_argument("model_path", help="local checkpoint directory or HF model ID")
    values.add_argument("--backend", choices=["paged", "hf"], default="paged")
    values.add_argument("--device", default="cuda")
    values.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    values.add_argument("--host", default="127.0.0.1")
    values.add_argument("--port", type=int, default=8000)
    values.add_argument("--block-size", type=int, default=16)
    values.add_argument("--kv-cache-num-blocks", type=int, default=256)
    values.add_argument("--max-num-seqs", type=int, default=4)
    values.add_argument("--max-num-batched-tokens", type=int, default=256)
    values.add_argument("--max-model-len", type=int, default=1024)
    values.add_argument("--max-prefill-chunk-size", type=int, default=128)
    values.add_argument(
        "--enable-chunked-prefill", action=argparse.BooleanOptionalAction,
        default=True,
    )
    values.add_argument("--enable-prefix-caching", action="store_true")
    values.add_argument("--enable-spec-decode", action="store_true")
    loading = values.add_mutually_exclusive_group()
    loading.add_argument("--load-in-4bit", action="store_true", help="HF NF4 only")
    loading.add_argument("--load-in-8bit", action="store_true", help="HF INT8 only")
    return values


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if not 1 <= args.port <= 65535:
        arguments.error("--port must be between 1 and 65535")
    if args.backend == "hf" and args.enable_prefix_caching:
        arguments.error("physical prefix caching requires --backend paged")
    if args.backend == "paged" and (args.load_in_4bit or args.load_in_8bit):
        arguments.error("bitsandbytes loading requires --backend hf")

    from nanovllm_pp import LLM, SamplingParams
    from nanovllm_pp.server.app import start_server

    engine = LLM(
        args.model_path, model_backend=args.backend, device=args.device,
        dtype=args.dtype, block_size=args.block_size,
        kv_cache_num_blocks=args.kv_cache_num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_spec_decode=args.enable_spec_decode,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
    )
    start_server(engine, SamplingParams, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
