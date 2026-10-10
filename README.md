# NanoVLLM++

A readable, single-GPU LLM inference runtime: continuous batching, chunked
prefill, physical KV pages and block tables, prefix reuse, and greedy n-gram
speculation. The CUDA backend runs a custom Qwen3 transformer with Triton KV
writes and **direct paged decode attention**. A Hugging Face backend provides
a compatibility/reference path and optional quantized loading.

This is an inspectable systems prototype, **not a production vLLM replacement**.
It has no CUDA graphs, distributed execution, preemption/swap, or fused paged
prefill kernel. Prefill gathers the physical context into PyTorch SDPA; decode
reads noncontiguous pages directly through device block tables.

[Architecture](docs/DESIGN.md) · [Usage / HTTP API](docs/USAGE.md) ·
[Reproduce results](docs/REPRODUCING.md) · [Raw evidence](benchmark_results/README.md)

## Results

### Controlled serving experiments — October 6, 2026

Qwen3-0.6B BF16, RTX 5050 Laptop GPU (7,708 MiB visible), PyTorch 2.10.0+cu130,
Transformers 5.8.1, Triton 3.6.0. Values are **medians of three warmed trials**;
model loading and kernel warmup are excluded. Laptop clocks/power were not
locked. These are direct-engine measurements, not HTTP or production capacity.

Same runner, 12 mixed requests (32/256/512 prompt tokens, 16/32 output tokens),
all submitted together; prefix reuse and speculation disabled:

| Scheduling width | Output tokens/s | TTFT p50 / p95 (ms) | TPOT p95 (ms) | E2E p95 (ms) |
|---|---:|---:|---:|---:|
| 1 | 57.969 | 2,648.1 / 4,867.5 | 17.538 | 5,197.4 |
| 4 | 155.522 | 730.6 / 1,475.9 | 21.459 | 1,890.3 |
| 8 | 222.731 | 212.9 / 776.9 | 23.782 | 1,305.1 |
| 8, Poisson arrivals at 4 requests/s | 90.094 | 77.4 / 158.2 | 21.361 | 756.2 |

Width eight delivers **3.84× aggregate throughput** versus width one, while
per-request TPOT increases. Width is work items per scheduler iteration, not
total live requests. Eleven of twelve greedy completions match the width-one
baseline in every trial; bitwise batch invariance is not claimed. Arrival-paced
throughput includes idle time.

Physical prefix reuse, 12 requests with a shared prefix + 32-token unique suffix,
16 output tokens, concurrency four:

| Shared prefix | Cache state | Reused prompt tokens | TTFT p50 / p95 (ms) | Output tokens/s |
|---|---|---:|---:|---:|
| 64 | Off | 0 | 34.725 / 36.191 | 235.379 |
| 64 | Warm | 768 | 40.049 / 43.301 | 216.710 |
| 256 | Off | 0 | 139.124 / 151.483 | 170.994 |
| 256 | Warm | 3,072 | 34.153 / 35.381 | 235.012 |
| 256 | Cold start | 2,048 | 34.635 / 137.018 | 207.329 |
| 512 | Off | 0 | 261.184 / 264.416 | 128.175 |
| 512 | Warm | 6,144 | 35.108 / 37.209 | 235.875 |

At 256 tokens, warm reuse cuts median TTFT **75.45%** and avoids exactly
3,072 prefill tokens. All 12 completions match cache-off in every trial at all
three prefix lengths. The 64-token case has **no consistent benefit**; cold
start still pays the first prefix computation. A separate 72-page pressure
case reaches **70/72 live pages (97.22%)**, completes at 93.932 output tokens/s,
and releases every request reference. Both width and pool size change in that
case: it tests capacity, not an isolated memory-size speed effect.

[Full methodology and correctness caveats](docs/RUNTIME_EVIDENCE.md) ·
[Compact CSV](benchmark_results/runtime_evidence/summary.csv) ·
[All trials, token IDs and traces](benchmark_results/runtime_evidence/results.json)

### Quantized loading — separate HF experiment

Same checkpoint/GPU, but Python 3.12.12, PyTorch 2.12.0+cu130, Transformers
5.11.0, Triton 3.7.0, bitsandbytes 0.49.2. Three prompts × 12 forced output
tokens, three repeats; **do not compare these rates to the serving table**.

| Format | Loaded CUDA allocation (GiB) | Output tokens/s | Exact BF16 matches / 3 | Fixed-text mean token NLL |
|---|---:|---:|---:|---:|
| BF16 | 1.110 | 75.25 | 3 | 4.8406 |
| bitsandbytes NF4 | 0.502 | 39.26 | 0 | 4.8781 |
| bitsandbytes INT8 | 0.732 | 14.01 | 2 | 4.9471 |
| Custom INT8 storage | 0.705 | 27.61 | 3 | 4.9412 |
| Custom packed INT4 storage | 0.497 | 17.53 | 0 | 5.5210 |

NF4 saves **54.8% loaded allocation**, but generation is **1.92× slower** here.
Allocation is measured after loading, before inference: not checkpoint size,
driver memory, or the paged KV pool. Custom INT8/INT4 modules dequantize full
weights before floating matmul; they are storage prototypes, not fused integer
kernels. NLL covers only 75 teacher-forced positions, not corpus quality.

[Storage/compute boundaries and commands](docs/QUANTIZATION_SCOPE.md) ·
[Raw loading and fidelity results](benchmark_results/runtime_evidence/quantized_loading.json)

### Earlier standard-workload sweep — August 24, 2026

Historical, single warmed runs with different shapes/settings; **not directly
comparable to the October medians**. Retained for workload coverage:

| Workload (input/output tokens; concurrency) | Output tokens/s | TTFT p50 (ms) |
|---|---:|---:|
| Decode-heavy (32/128; 8) | 520.7 | 45.5 |
| Balanced (256/64; 8) | 423.3 | 219.4 |
| Prefill-heavy (1,024/16; 4) | 89.2 | 463.1 |
| ShareGPT (16 variable-length requests; 8) | 393.6 | 53.0 |

The historical balanced HF reference measured 74.0 output tokens/s versus
423.3 for paged execution. This compares **whole implementations** (serial HF
dispatch versus custom batched execution), not paging alone. No benchmark
against production vLLM is claimed.

[Concurrency, arrival-rate, prefix and ShareGPT details](docs/STANDARD_BENCHMARKS.md)

## Quick start

Python 3.10+; the paged backend requires NVIDIA CUDA access and Triton on a
supported platform. Use an isolated environment and a GPU-supported PyTorch
build. [Recorded environments and checkpoint hashes](docs/REPRODUCING.md) are
available for reproduction; loose requirements do not imply testing every
version or GPU.

```bash
git clone https://github.com/AmanPandey28/nano_vllm_pp.git
cd nano_vllm_pp
python -m pip install -e '.[paged,server,test]'
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
python -m nanovllm_pp.benchmarks.gpu_smoke
python examples/generate.py ./Qwen3-0.6B
```

Weights are downloaded separately, not committed. Explicit `device="cuda"`
fails if the process cannot see the GPU. A working host `nvidia-smi` does not
guarantee that a container can access `/dev/nvidia*`.

```python
from nanovllm_pp import LLM, SamplingParams

llm = LLM(
    "./Qwen3-0.6B", device="cuda", dtype="bfloat16", model_backend="paged",
    block_size=16, kv_cache_num_blocks=256, max_num_seqs=4,
    max_num_batched_tokens=256, max_model_len=1024,
    enable_chunked_prefill=True, max_prefill_chunk_size=128,
    enable_prefix_caching=True,
)
outputs = llm.generate("The capital of France is", SamplingParams(temperature=0, max_tokens=32))
print(outputs[0]["generated_text"])
```

For terminal output **and metrics**, choose one of three matched demos:

```bash
python -m nanovllm_pp.benchmarks.live_paged_baseline --model ./Qwen3-0.6B
python -m nanovllm_pp.benchmarks.live_paged_optimized --model ./Qwen3-0.6B
python -m nanovllm_pp.benchmarks.live_hf_reference --model ./Qwen3-0.6B
```

Each prints a stop-aware sample, then a fixed-output benchmark. Baseline has
chunking/prefix reuse off; optimized enables both; HF has chunking but no
physical prefix reuse. This combined demo is not a one-variable ablation.

## HTTP server

```bash
python -m nanovllm_pp.server ./Qwen3-0.6B --enable-prefix-caching
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","prompt":"A KV cache stores","max_tokens":16,"temperature":0,"stream":true}'
curl http://127.0.0.1:8000/metrics
```

One process owns one model. Requests can join later iterations; model calls
are serialized under an engine lock. Endpoints are OpenAI-shaped, not full
wire compatibility; `model` is a response label, not multi-model routing. No
authentication is provided: default bind is localhost.
[Flags, endpoints, lifecycle and metric definitions](docs/USAGE.md).

## Validate and reproduce

```bash
python -m pytest -q -o faulthandler_timeout=30
python -m ruff check nanovllm_pp tests examples
python -m nanovllm_pp.benchmarks.validate_paged_backend ./Qwen3-0.6B
python -m nanovllm_pp.benchmarks.bench_runtime_evidence ./Qwen3-0.6B \
  --output-dir runs/runtime_evidence --repeats 3 --num-prompts 12 --seed 17
```

CPU tests use model doubles and require no checkpoint download. GPU tests skip
without CUDA/Triton; a CPU-green CI run does not validate GPU execution. The
suite covers randomized allocator audits, attention oracles, scheduler state,
and failure/stream cleanup. New runs go to ignored `runs/`, leaving published
evidence unchanged. [October 10 validation](docs/VALIDATION.md): **113 tests
passed with CUDA visible**, plus server/install/package checks.
[Complete reproduction workflow](docs/REPRODUCING.md).

## Architecture and repository map

```text
Caller / HTTP -> LLMEngine.step() -> Scheduler -> ModelRunner -> model + KV pool
                                      |              ^
                                      v              |
                                 BlockManager -------+
```

The scheduler chooses work under token/sequence budgets. The block manager
owns page identity, references and prefix hashes. The paged runner owns tensor
bytes and builds device metadata; model/kernels execute computation.
Postprocessing commits valid KV, updates state, and releases finished requests.

| Location | Responsibility |
|---|---|
| [engine/](nanovllm_pp/engine/) | Request state, allocation, scheduler, engine loop and HF runner |
| [paged/](nanovllm_pp/paged/) | Custom Qwen3, global KV pool, device block tables and Triton kernels |
| [scheduling/](nanovllm_pp/scheduling/) | Token-budget policy and chunked prefill |
| [spec_decode/](nanovllm_pp/spec_decode/) | N-gram proposal and exact greedy target verification |
| [quantization/](nanovllm_pp/quantization/) | Per-channel INT8 / packed INT4 storage prototypes |
| [server/](nanovllm_pp/server/) / [observability/](nanovllm_pp/observability/) | HTTP/SSE, cleanup, traces and metrics |
| [benchmarks/](nanovllm_pp/benchmarks/) / [examples/](examples/) | Workload harnesses and runnable entry points |
| [tests/](tests/) / [docs/](docs/README.md) | Correctness contracts, API and reproduction |
| [benchmark_results/](benchmark_results/README.md) | Dated machine-readable measurement evidence |

The paged backend supports standard full-attention `Qwen3ForCausalLM` models
with even head dimension ≤256. Unsupported families, sliding windows,
non-default RoPE and bitsandbytes loading are rejected. Only full, physically
computed prefix pages are shared; mutable partial pages are never aliased.
Speculation is greedy n-gram lookahead, **not a separate draft model**;
stochastic sampling and stop strings use ordinary one-token decoding.
Ordinary mixed iterations use at most two forwards; per-request speculative
verification can add forwards. Pool exhaustion reports no progress rather
than silently swapping.

[Detailed design / invariants](docs/DESIGN.md) ·
[Design lineage / references](docs/REFERENCES.md) · [MIT license](LICENSE)
