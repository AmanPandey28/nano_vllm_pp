# Technical references and implementation lineage

NanoVLLM++ is an independent educational implementation. Its page-pool and
attention design was cross-checked against the following primary sources:

- [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180) for the
  operating-system-inspired logical-to-physical KV mapping and serving goals.
- [vLLM automatic prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
  for full-block identity, chained prefix hashes, references, and eviction.
- [vLLM paged-attention kernel documentation](https://docs.vllm.ai/en/latest/design/paged_attention/)
  for online softmax and physical cache access considerations.
- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
  (MIT) as a compact comparison point. In particular, its
  [`attention.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/attention.py)
  demonstrates a Triton slot-write kernel and FlashAttention block-table APIs,
  while its
  [`model_runner.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/model_runner.py)
  constructs device slot mappings and block tables.
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
  and the [Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
  for the reference prefill operation and online-softmax kernel structure.

This project makes different trade-offs from GeeeekExplorer/nano-vllm. It does
not require FlashAttention—which is important for the tested RTX 5050/SM120
environment. It uses a purpose-built Triton decode kernel, keeps a PyTorch
correctness oracle beside it, batches flattened Qwen3 projections, and retains
the original HF backend for differential validation. It does not yet implement
CUDA graphs or tensor parallelism.

Relevant accelerator-integration comparisons are:

- [Qualcomm vLLM QAIC](https://github.com/qualcomm/vllm-qaic), a vLLM backend
  plugin with PyTorch eager and ahead-of-time modes for Cloud AI accelerators.
- [Qualcomm efficient-transformers](https://github.com/quic/efficient-transformers),
  which converts Hugging Face checkpoints to inference-ready forms for
  Qualcomm Cloud AI accelerators.

Those Qualcomm repositories are references only; NanoVLLM++ does not contain
their code and does not claim QAIC hardware support.
