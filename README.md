# NanoVLLM++

NanoVLLM++ is a small, readable LLM inference-runtime prototype. It exposes the
moving parts behind vLLM-style serving: request state, prefill/decode
separation, iteration-level scheduling, fixed-size KV pages, block tables,
physical prefix reuse, chunked prefill, exact greedy n-gram speculation,
sampling, observability, quantization prototypes, and an OpenAI-shaped API.

Two execution backends make the boundary testable. `model_backend="hf"` is the
general Hugging Face compatibility/reference path. `model_backend="paged"` is
a CUDA Qwen3 path with a custom transformer, one preallocated global K/V pool,
Triton page writes, and a Triton decode kernel that follows device block tables
directly. Paged prefill reads the same physical pool but materializes each
logical context for PyTorch SDPA, favoring clarity over a production fused
varlen kernel.

This is intentionally not a production vLLM replacement. It has no CUDA graph
capture, preemption/swap, distributed parallelism, fused paged-prefill kernel,
or broad model-family support.

## Feature truth table

| Capability | Status | Important boundary |
|---|---|---|
| Prefill/decode KV reuse | Both backends | HF DynamicCache or global physical page pool |
| Chunked prefill | Implemented | Cache is carried across chunks; waiting prompts rotate fairly |
| Iteration-level scheduling | Implemented | Paged backend tensor-batches prefill and decode groups; a mixed iteration uses at most two forwards |
| Physical KV pages/block tables | Paged backend | Triton scatter writes; direct page-table decode reads; readable gathered SDPA prefill |
| Prefix caching | Paged backend | Full physically valid pages only; chained hashes + refcounts |
| N-gram speculative decode | Greedy sampling | Exact target verification and logical commit of only the accepted physical prefix |
| Top-k/top-p/min-p/seed/logprobs/stops | Implemented | Speculation falls back to one-token steps for stop strings and stochastic sampling |
| INT8/packed INT4 modules | Educational implementation | Dequantizes weights before matmul, so it saves storage but is not a fused speed kernel |
| bitsandbytes NF4/INT8 | Optional CUDA path | Requires the `quantization` extra and CUDA |
| FastAPI + SSE | Implemented, OpenAI-shaped | Not claimed as wire-compatible with every OpenAI field |
| CUDA graphs / `torch.compile` | Not implemented | `enforce_eager=False` is rejected explicitly |
| Tensor/pipeline parallelism | Not implemented | Single Python process and model instance |

## Install and test

```bash
pip install -e ".[test]"
pytest -q
ruff check nanovllm_pp tests
```

Optional HTTP and bitsandbytes dependencies:

```bash
pip install -e ".[server]"
pip install -e ".[quantization]"
pip install -e ".[paged]"
```

## CUDA access and fail-fast validation

Performance runs should request CUDA explicitly. This prevents a restricted
container or agent sandbox from silently running a GPU benchmark on the CPU:

```bash
nvidia-smi
python -m nanovllm_pp.benchmarks.gpu_smoke
```

If `nvidia-smi` works in a normal terminal but the smoke command reports that
CUDA is unavailable in an automation environment, that process does not have
access to `/dev/nvidia*`. Run it with host GPU/device access; reinstalling
PyTorch inside the isolated process will not expose a hidden device.

The engine accepts `device="auto"`, `"cpu"`, `"cuda"`, or `"cuda:N"` and
`dtype="auto"`, `"float32"`, `"float16"`, or `"bfloat16"`. CUDA benchmarks in
this repository request `cuda` and fail clearly when it is not visible. On
CUDA, automatic dtype selection prefers BF16 when the device supports it and
otherwise uses FP16.

Download or point to a local Hugging Face causal LM. Qwen3-0.6B is the model
used in the design examples:

```bash
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
```

## Basic usage

```python
from nanovllm_pp import LLM, SamplingParams

llm = LLM(
    "./Qwen3-0.6B",
    device="cuda",
    dtype="auto",
    model_backend="paged",
    block_size=16,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    enable_spec_decode=True,
)
params = SamplingParams(temperature=0.0, max_tokens=32, logprobs=5)

outputs = llm.generate(
    ["The capital of France is", "Machine learning is"],
    params,
)
for output in outputs:
    print(output["generated_text"], output["finish_reason"])
```

`generate()` is synchronous. For iteration-level admission, call
`add_request()`, repeatedly call `step()`, read `get_output()` after completion,
then call `forget_request()` to release retained output state.

The paged backend currently accepts standard full-attention
`Qwen3ForCausalLM` checkpoints with even `head_dim <= 256`. It rejects
unsupported model families, non-default RoPE, sliding-window attention,
bitsandbytes loading, and CPU execution instead of silently falling back.
`kv_cache_num_blocks=N` is a deterministic demo/test override; otherwise the
engine sizes the physical pool from live CUDA headroom.

## Serving

```python
from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.server.app import start_server

llm = LLM("./Qwen3-0.6B", device="cuda", enable_chunked_prefill=True)
start_server(llm, SamplingParams, port=8000)
```

```bash
curl http://localhost:8000/health
curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","prompt":"Hello","max_tokens":16,"temperature":0,"stream":true}'
```

The app uses an async lock around the non-thread-safe engine and yields between
engine steps. Concurrent requests can therefore join later scheduler
iterations; model calls themselves remain serialized. `/metrics` reports
queue depth, logical-block usage, engine steps, phase timing, and TTFT/TPOT/E2E
latency summaries for requests observed by the server.

## Tutorials and benchmarks

```bash
python nanovllm_pp/tutorials/01_naive_generation.py ./Qwen3-0.6B
python nanovllm_pp/tutorials/02_kv_cache.py ./Qwen3-0.6B
python nanovllm_pp/tutorials/03_engine_basic.py ./Qwen3-0.6B

python nanovllm_pp/benchmarks/bench_baseline.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_chunked_prefill.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_spec_decode.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_nf4.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_quantization.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_profiling.py ./Qwen3-0.6B
python -m nanovllm_pp.benchmarks.bench_paged_backend ./Qwen3-0.6B
python -m nanovllm_pp.benchmarks.bench_serving ./Qwen3-0.6B \
  --dataset-name random --random-input-len 256 --random-output-len 64
python -m nanovllm_pp.benchmarks.validate_paged_backend ./Qwen3-0.6B
```

Benchmarks warm the model and synchronize CUDA around timed regions, but they
remain workload examples rather than universal performance claims. Record the
model revision, hardware/power mode, software versions, sample distribution,
request shapes, precision, and output correctness before comparing runs.

## Repository map

```text
nanovllm_pp/
├── engine/           sequence, logical blocks, scheduler, model runner, engine
├── paged/            Qwen3 model, global KV runner, Triton page kernels
├── scheduling/       token budget and mixed chunked-prefill policy
├── spec_decode/      n-gram proposer, greedy verifier, integrated manager
├── quantization/     per-channel INT8 and packed INT4 teaching modules
├── observability/    phase/request metrics and JSONL tracing
├── server/           FastAPI completion/chat/SSE surface
├── benchmarks/       reproducible benchmark entry points
└── tutorials/        three progressive executable examples
tests/                deterministic CPU tests with a tiny causal-model double
```

See [docs/DESIGN.md](docs/DESIGN.md) for the architecture, implementation
invariants, known limitations, and validation plan. The exact RTX 5050
environment, synchronized measurements, parity results, and caveats are in
[docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md). Design lineage and external
comparison points are recorded in [docs/REFERENCES.md](docs/REFERENCES.md).
The fixed-shape, concurrency, ShareGPT, arrival-rate, and physical-prefix
results are in [docs/STANDARD_BENCHMARKS.md](docs/STANDARD_BENCHMARKS.md), with
machine-readable JSON under `benchmark_results/`.

## License

MIT
