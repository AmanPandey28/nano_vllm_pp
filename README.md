# NanoVLLM++

A minimal, readable LLM inference engine that implements the core ideas behind production systems like vLLM. Every component exists to teach a specific runtime concept.

## What it implements

| Component | Concept |
|-----------|---------|
| Prefill/decode separation | Why KV caches eliminate O(n²) recomputation |
| Paged KV cache | Logical-to-physical block mapping, ref-counted sharing |
| Prefix caching | Chained-hash block reuse for repeated prompts |
| Continuous batching | Dynamic request admission and completion |
| Chunked prefill | Token-budget scheduling, decode interleaving |
| Speculative decoding | N-gram proposer + greedy verifier |
| Weight-only quantization | Per-channel INT8/INT4 with memory benchmarks, NF4 via bitsandbytes |
| OpenAI-compatible serving | FastAPI, SSE streaming, /metrics |

## Quickstart

```bash
pip install -e .
# or: pip install -e ".[server]"  # for the HTTP API

# Download a model (Qwen3-0.6B recommended)
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
```

## Basic usage

```python
from nanovllm_pp import LLM, SamplingParams

llm = LLM("./Qwen3-0.6B", enforce_eager=True)

params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(
    ["The capital of France is", "Machine learning is"],
    params,
)

for out in outputs:
    print(out["generated_text"])
```

## Tutorials

```bash
# 01 — naive generation (O(n^2) baseline)
python nanovllm_pp/tutorials/01_naive_generation.py ./Qwen3-0.6B

# 02 — KV cache (prefill/decode)
python nanovllm_pp/tutorials/02_kv_cache.py ./Qwen3-0.6B

# 03 — full engine pipeline
python nanovllm_pp/tutorials/03_engine_basic.py ./Qwen3-0.6B
```

## Benchmarks

```bash
python nanovllm_pp/benchmarks/bench_baseline.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_chunked_prefill.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_spec_decode.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_quantization.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_nf4.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_profiling.py ./Qwen3-0.6B
```

## Serving

```python
from nanovllm_pp.server.app import start_server
from nanovllm_pp import LLM, SamplingParams

llm = LLM("./Qwen3-0.6B", enforce_eager=True)
start_server(llm, SamplingParams, port=8000)
```

```bash
# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","prompt":"Hello","max_tokens":16,"temperature":0.0}'
```

## Quantized inference (NF4)

```python
from nanovllm_pp import LLM, SamplingParams

# Load with NF4 quantization — ~50% memory reduction
llm = LLM("./Qwen3-0.6B", load_in_4bit=True)
```

```python
from nanovllm_pp import LLM, SamplingParams

llm = LLM("./Qwen3-0.6B", enforce_eager=True)
llm.enable_profiling()

outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=12))
llm.profile_report()
# Prints per-phase timing breakdown (schedule/prefill/decode/postprocess)
# and per-request latency percentiles (TTFT, TPOT, E2E)
```

## Structure

```
nanovllm_pp/
├── engine/           Core runtime (sequence, block_manager, scheduler, runner, engine)
├── scheduling/       Token budget, chunked prefill policies
├── spec_decode/      N-gram proposer, verifier, speculation manager
├── quantization/     INT8/INT4 per-channel weight quantization
├── observability/    Engine profiler, request tracker, trace events
├── server/           FastAPI OpenAI-compatible HTTP API
├── benchmarks/       Baseline, chunked prefill, spec decode, quantization, profiling
└── tutorials/        Naive generation, KV cache, full engine
```

## Design doc

See [docs/DESIGN.md](docs/DESIGN.md) for a detailed walkthrough of every component.

## License

MIT
