# Quantization: storage, arithmetic, and measured boundaries

Quantized loading is supported through the **HF backend**. The custom paged
Qwen3 backend rejects bitsandbytes loading; NF4 memory numbers do not describe
the paged runner or its preallocated global KV pool. BF16 model and KV compute
dtype selection lives in `engine/llm_engine.py`; quantized weights do not
automatically make the KV cache four-bit.

All benchmark formats start from the same local BF16 `model.safetensors`, not
a prequantized checkpoint. On-disk input format therefore remains BF16 in this
experiment. Embeddings, norms, and the output head retain BF16 where not
converted. The loader explicitly passes the requested dtype to nonquantized
modules as well as setting the NF4 compute dtype.

| Path | In-memory linear weights | Scale granularity | Dequantization / arithmetic / operator | Status |
|---|---|---|---|---|
| BF16 | BF16 tensors | None | Floating GEMM/GEMV (`nn.Linear` / PyTorch); BF16 operands | Reference loading |
| bitsandbytes NF4 | Two codebook indices per uint8 plus quantization state; double-quantized scales | Library block quantization, normally 64 weights/block; nested scale quantization | `Linear4bit` / `bnb.matmul_4bit`: library selects GEMV or dequantization + floating linear; configured BF16 compute, not a project INT4 GEMM | Optional library integration |
| bitsandbytes INT8 | INT8 linear tensors plus scale/outlier state | LLM.int8 row-wise state with activation quantization/outlier handling | `Linear8bitLt` / library matmul, INT8 path plus higher-precision outlier work; activations may be cast from BF16 to FP16 | Optional library integration |
| Custom INT8 | INT8 buffers, FP32 scale per output channel | Symmetric scale `max(abs(row))/127` | Rebuild the full FP32 weight, cast to input BF16, then `x @ weight.T`; floating GEMM/GEMV | Storage prototype |
| Custom packed INT4 | Two signed four-bit values per uint8, FP32 row scales; zero-nibble padding for odd input width | Symmetric scale `max(abs(row))/7` | Unpack entire matrix to FP32, scale and cast to BF16, then floating matmul | Packed storage prototype |

For custom INT4, the encoded nibble is `q+8`; low/high nibbles interleave input
columns. Padding nibble 8 represents zero and is sliced away on unpack. These
custom modules save resident storage but allocate temporary floating weights
on every forward. Calling them fused INT4 compute or asserting a speedup
would misrepresent their implementation. No fused W4A16 kernel is integrated
into this runtime.

The NF4/LLM.int8 distinction and library interfaces are documented in the
[official Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
Dispatch is shape/version dependent; this table names the entry points rather
than claiming every call uses one fixed native kernel.

## Reproduce the loading experiment

The recorded run used the existing base environment with Python 3.12.12,
PyTorch 2.12.0+cu130, Transformers 5.11.0, Triton 3.7.0, bitsandbytes 0.49.2,
and Accelerate 1.14.0 on the RTX 5050. This is a separate, internally matched
quantization comparison from the serving experiment's `mlsys` environment;
cross-environment latency comparisons are not made.

```bash
python -m nanovllm_pp.benchmarks.bench_loading_evidence \
  ./Qwen3-0.6B \
  --output runs/quantized_loading.json \
  --repeats 3 --max-tokens 12
```

Use the separate pinned environment in [REPRODUCING.md](REPRODUCING.md).
For another environment install the optional extra, then use its Python:

```bash
pip install -e '.[quantization]'
python -m nanovllm_pp.benchmarks.bench_loading_evidence ./Qwen3-0.6B
```

The script loads each format separately, releases it with garbage collection
and `empty_cache()`, subtracts the pre-load allocated baseline, synchronizes
CUDA, warms the full workload, and measures three repeats. Each of three
prompts generates 12 forced greedy tokens. Loading/conversion is excluded
from serving timing and reported separately.

**Loaded CUDA allocation** is the difference in `torch.cuda.memory_allocated()`
after loading/conversion versus before loading, before inference. It includes
live CUDA quantization metadata and allocator rounding. It is not serialized
checkpoint size, driver-reserved memory, `nvidia-smi` usage, or peak generation
memory. The artifact separately records peak load/generation allocation and
counted parameter/buffer bytes. The latter counter does **not** include every
bitsandbytes quantization-state tensor held outside registered buffers.

## Results

| Format | Loaded CUDA allocation GiB | Output tok/s | Median generation s | Exact BF16 completion matches / 3 | Fixed-text mean token NLL |
|---|---:|---:|---:|---:|---:|
| BF16 | 1.110 | 75.25 | 0.478 | 3 | 4.8406 |
| NF4 | 0.502 | 39.26 | 0.917 | 0 | 4.8781 |
| bitsandbytes INT8 | 0.732 | 14.01 | 2.570 | 2 | 4.9471 |
| Custom INT8 storage | 0.705 | 27.61 | 1.304 | 3 | 4.9412 |
| Custom packed INT4 storage | 0.497 | 17.53 | 2.053 | 0 | 5.5210 |

NF4 reduces loaded allocation by about **54.8%**, but takes about **1.92×**
the BF16 generation time on this workload. The older 1.11→0.53 figure in
`GPU_VALIDATION.md` is a historical observation from an older environment;
the current reproduced result is **1.11→0.50 GiB loaded CUDA allocation**.
Values are binary GiB, not decimal GB.

Fidelity diagnostics use 75 teacher-forced next-token predictions across three
fixed synthetic texts, with mean token NLL and last-position logit cosine/max
error versus BF16. The texts and every diagnostic are in
[`quantized_loading.json`](../benchmark_results/runtime_evidence/quantized_loading.json).
All outputs were finite. This small probe is not corpus perplexity, retrieval
accuracy, or a guarantee that downstream quality is unchanged. Exact greedy
matches are sensitive to argmax differences and do not replace quality
evaluation. The packed INT4 prototype's larger NLL error remains visible.

## Code and test map

- `engine/llm_engine.py`: validated BF16/NF4/INT8 loader dispatch, dtype, and backend restrictions.
- `quantization/int8_weight_only.py`: per-channel quantization, nibble packing, full-weight dequantization, conversion, and registered-storage counter.
- `benchmarks/bench_loading_evidence.py`: fixed prompts/texts, memory boundaries, warm timing, logits/NLL, exact output comparisons, and JSON.
- `tests/test_quantization.py` and `tests/test_cuda.py`: odd-width unpacking, error bound, buffer storage, and actual CUDA execution.

The primary measured runtime optimization remains same-runner batching and
physical prefix reuse. Quantization is an explicit memory/latency/fidelity
tradeoff in this runtime.
