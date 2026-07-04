# NanoVLLM++ — Design Document

A from-scratch LLM inference engine implementing the core runtime concepts behind
production systems like vLLM and SGLang. Uses Qwen3-0.6B as the primary test
model and facebook/opt-125m for rapid iteration.

---

## 1. Architecture

```
nanovllm_pp/
├── engine/
│   ├── sequence.py          Per-request state machine + block table
│   ├── block_manager.py     Paged KV cache, ref counting, prefix hashing
│   ├── model_runner.py      Model loading, prefill/decode, sampling
│   ├── scheduler.py         Continuous batching, admission control
│   └── llm_engine.py        Main runtime loop
├── scheduling/
│   ├── token_budget.py      Token-level scheduling budget
│   └── chunked_prefill.py   Mixed prefill/decode batch builder
├── spec_decode/
│   ├── proposer.py          NGramProposer + GreedyVerifier
│   └── manager.py           SpecDecodeManager with metrics
├── quantization/
│   └── int8_weight_only.py  Per-channel INT8/INT4 quantization
├── observability/
│   ├── profiler.py          Engine phase + request latency profiling
│   └── trace.py             JSONL event tracing
├── server/
│   └── app.py               FastAPI OpenAI-compatible API
├── benchmarks/              6 benchmark scripts
├── tutorials/               3 progressive tutorial scripts
├── config.py                EngineConfig + ModelConfig
├── sampling_params.py       Per-request sampling configuration
└── llm.py                   Public API
```

---

## 2. Core Components

### 2.1 Sequence (`engine/sequence.py`)

Each generation request is a `Sequence` tracking:
- `token_ids`: full list of prompt + generated tokens
- `block_table`: list of physical KV block IDs mapping logical positions to GPU memory
- `num_computed_tokens`, `num_generated_tokens`, `num_cached_tokens`: progress counters
- `status`: WAITING → RUNNING → FINISHED (or PREEMPTED)
- `enqueue_time`, `first_token_time`, `completion_time`: latency tracking

### 2.2 BlockManager (`engine/block_manager.py`)

Paged KV cache memory manager. The KV cache is split into fixed-size blocks
(default 256 tokens each). Each sequence gets a block table mapping logical
positions to physical block IDs.

**Core operations:**
- `allocate(seq)`: allocate blocks for a full prompt, attempting prefix cache reuse
- `deallocate(seq)`: release blocks, decrement ref counts
- `may_append(seq)`: ensure a block exists for the next generated token
- `compute_block_hash()`: chained hashing for prefix detection

**Reference counting:** blocks shared between sequences via prefix caching are
safe because deallocation only frees a block when ref_count reaches zero.

**GPU memory estimation:**
```
kv_per_token  = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
kv_per_block  = kv_per_token × block_size
available     = total_gpu × utilization − model_memory − cuda_overhead(512MB)
num_kv_blocks = available / kv_per_block
```

For Qwen3-0.6B (28 layers, 8 KV heads, 128 head_dim, bf16):
- kv_per_token = 2 × 28 × 8 × 128 × 2 = 114,688 bytes (~112 KB)
- kv_per_block = 114,688 × 256 ≈ 28 MB

### 2.3 ModelRunner (`engine/engine/model_runner.py`)

Loads the HF model and executes prefill/decode passes using
`past_key_values` for KV caching.

**Prefill:** processes prompt tokens in one or more chunks. Returns the
first generated token after the full prompt is processed. Stores
`past_key_values` on the sequence for subsequent decode steps.

**Decode:** feeds a single token with the cached `past_key_values`.
Updates the cache after each forward.

**Sampler:** temperature scaling, top-k, top-p, greedy/temperature sampling.

### 2.4 Scheduler (`engine/scheduler.py`)

Maintains two FIFO queues: `waiting` (unprefilled requests) and `running`
(in-progress requests). The schedule loop:

1. If `waiting` has items → admit sequences for prefill (checking budget + blocks)
2. If only `running` has items → schedule one decode step per sequence

With chunked prefill enabled, the `MixedBatchScheduler` interleaves decode
tokens with prefill chunks under a shared token budget.

### 2.5 LLMEngine (`engine/llm_engine.py`)

Initialization flow:
1. Parse kwargs → EngineConfig
2. Load tokenizer
3. Load model (FP16, or NF4/INT8 via bitsandbytes)
4. Build ModelConfig from HF config (handles multi-modal `text_config`)
5. Estimate KV cache blocks from GPU memory
6. Create BlockManager → ModelRunner → Scheduler

Main loop in `step()`: schedule → model_runner.run() → postprocess.

---

## 3. Chunked Prefill

Splits long prompts into chunks (default 512 tokens) under a shared token
budget (`max_num_batched_tokens`). Decode steps get priority on the budget
to minimize inter-token latency for existing requests.

**Algorithm:**
1. Phase A: iterate running queue, schedule 1 decode token per sequence
2. Phase B: iterate waiting queue, schedule prefill chunks with remaining budget
3. New sequences reserve full prompt blocks on first admission
4. A sequence exits waiting when its full prompt has been prefilled

---

## 4. Speculative Decoding

**NGramProposer:** searches backward through generated tokens for a suffix
match (length 2–5). If found, proposes the tokens that followed that match.
No second model needed.

**GreedyVerifier:** for each proposed token, compares against the target
model's argmax prediction. Accept on match, reject and stop on mismatch.

**Metrics:** acceptance rate (accepted/proposed), target forwards saved,
empty proposal rate.

---

## 5. Quantization

### 5.1 Per-Channel INT8/INT4 (`quantization/int8_weight_only.py`)

Each output channel gets its own scale factor from max-abs calibration.
`Int8WeightOnlyLinear` wraps a quantized linear layer with on-the-fly
dequantization.

**Memory savings:** FP16=2B, INT8=1B, INT4=0.5B per weight.

### 5.2 NF4 via bitsandbytes (`load_in_4bit=True`)

NormalFloat 4-bit quantization using bitsandbytes. Uses the
information-theoretically optimal codebook for normally-distributed
weights. Double quantization compresses the per-block scales.

**Supported via:** `LLM(model_path, load_in_4bit=True)`

---

## 6. Profiling & Diagnostics

### 6.1 EngineProfiler (`observability/profiler.py`)

Instruments each `step()` call to measure wall-clock time per phase:
- **Schedule**: scheduler decision logic
- **Prefill**: model forward for prompt tokens
- **Decode**: model forward for single-token generation
- **Postprocess**: sequence state updates, block deallocation

### 6.2 RequestTracker

Tracks per-request lifecycle latency:
- **TTFT** (Time to First Token): enqueue → first generated token
- **TPOT** (Time Per Output Token): (completion − first token) / tokens generated
- **E2E**: total wall-clock time

### 6.3 `diagnose_slowdown()`

Automatic bottleneck detection combining engine phase timing and request percentiles:
- >50% schedule time → scheduler logic or block allocation issue
- >100ms avg prefill → enable chunked prefill
- >500ms p50 TTFT → queue backlog, check block availability
- >50ms p50 TPOT → decode is slow, consider quantization or speculation

---

## 7. Serving (`server/app.py`)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | GPU status, engine readiness |
| `GET /metrics` | Queue depths, KV block usage, total steps |
| `POST /v1/completions` | Text completions (stream + non-stream via SSE) |
| `POST /v1/chat/completions` | Chat messages → prompt template → completions |

---

## 8. Benchmark Results

All benchmarks run on NVIDIA RTX 5050 Laptop GPU (8GB VRAM).

### 8.1 Baseline: Naive vs KV Cache vs Engine (Qwen3-0.6B)

3 prompts, 16 tokens each, greedy decoding.

| Method | tok/s | Speedup |
|--------|-------|---------|
| Naive (recomputes full context) | 50.4 | 1.0× |
| KV Cache (prefill/decode) | 79.8 | 1.6× |
| **NanoVLLM++** (full pipeline) | **67.9** | **1.3×** |

Naive is O(n²) — each step reprocesses all accumulated tokens. KV cache
avoids recomputation. Engine adds scheduler/block overhead.

### 8.2 Baseline: Naive vs KV Cache vs Engine (OPT-125M)

Same workload on smaller model for comparison.

| Method | tok/s | Speedup |
|--------|-------|---------|
| Naive | 101.3 | 1.0× |
| KV Cache | 366.0 | 3.6× |
| **NanoVLLM++** | **310.8** | **3.1×** |

### 8.3 Chunked Prefill (Qwen3-0.6B)

1 long prompt (129 words) + 5 short prompts, 12 tokens each.

| Method | Time | tok/s |
|--------|------|-------|
| No chunking | 1.39s | 51.8 |
| **Chunked (512)** | **1.02s** | **70.6** |

1.36× speedup. Chunked prefill lets short prompts start while the
long prompt is still being processed.

### 8.4 Chunked Prefill (OPT-125M)

| Method | Time | tok/s |
|--------|------|-------|
| No chunking | 0.50s | 143.6 |
| **Chunked (512)** | **0.24s** | **299.2** |

### 8.5 Speculative Decoding — N-gram (Qwen3-0.6B)

Repetitive pattern prompt (country capitals), 12 tokens, greedy.

| Method | Time | Target Forwards | Accept Rate |
|--------|------|-----------------|-------------|
| Standard decode | 0.23s | 12 | — |
| **N-Gram spec decode** | **0.15s** | **12** | **0.75** |

12 tokens proposed, 9 accepted across 12 target forwards. 3 empty
proposals where no n-gram match was found.

### 8.6 INT8 Per-Channel Quantization (Qwen3-0.6B)

| Method | Weight Memory | Time | Output Correct |
|--------|---------------|------|----------------|
| FP16 | 1137 MB | 0.47s | "Paris. The capital of Italy is Rome" |
| **INT8** | **148 MB*** | **0.38s** | Same output |

*INT8 memory shown is remaining FP16 params (embeddings + lm_head + norms).
Quantized weights are stored as int8 buffers, not counted by
`named_parameters()`.

### 8.7 NF4 Quantization (Qwen3-0.6B)

| Method | GPU Memory | Time | tok/s |
|--------|-----------|------|-------|
| FP16 | 1.11 GB | 0.912s | 39.5 |
| **NF4** | **0.53 GB** | **0.998s** | **36.1** |
| INT8 (bitsandbytes) | 0.76 GB | 2.811s | 12.8 |

NF4 saves 52% GPU memory with 91% speed retention. Minor quality tradeoff
on this small model.

### 8.8 Profiling (Qwen3-0.6B)

Single prompt, 8 tokens generated.

**Engine phase breakdown:**
```
Phase           Avg (ms)     % of step
Schedule        0.009          0.0%
Prefill         46.010        76.5%
Decode          14.145        23.5%
Postprocess     0.005          0.0%
Bottleneck: prefill
```

**Request latency:**
```
Metric     p50
TTFT       368.1ms
TPOT       14.2ms
E2E        481.6ms
```

---

## 9. GPU Memory Analysis (Qwen3-0.6B)

```
Model:       604M params × 2 bytes (bf16) = 1.21 GB
KV per block: 2 × 28 layers × 8 heads × 128 dim × 2 bytes × 256 tokens = 28 MB/block
Total blocks: ~188 on 8GB GPU (after model + CUDA overhead)
Max context: 188 × 256 = 48,128 tokens
```

---

## 10. Debugging Notes

### OPT vs Qwen3 config keys
OPT uses `ffn_dim`, Qwen3 uses `intermediate_size`. Handled with `getattr()`
fallback chains.

### Initialization order
Model must be loaded before `_estimate_kv_blocks()` since estimation needs
`model.parameters()`. ModelRunner needs `block_manager` which needs
`num_kv_blocks`. Order: model → config → estimate → block_manager → runner.

### Gibberish output without KV cache
The decode step passes only the last token. Without `past_key_values`, the
model has zero context. Fix: use `use_cache=True` during prefill, store
`past_key_values` on the sequence, and reuse them during decode.

### Profiler tick placement
Model forward time was attributed to "sample" instead of prefill/decode.
Fix: place tick calls immediately before and after `model_runner.run()`,
not around `postprocess()`.

### TPOT calculation
`RequestTracker.tokens_generated` was always 0 because `record_token()` was
never called. Fix: pass `num_generated_tokens` from the sequence during
`record_completion()`.

### Multi-modal config support
Gemma 4 uses `Gemma4UnifiedConfig` with model parameters under `text_config`.
Fix: unwrap `text_config` in `_build_model_config()` before reading
architectural parameters.

---

## 11. Running

```bash
pip install -e ".[server]"
pip install bitsandbytes  # for NF4 quantization

# Tutorials
python nanovllm_pp/tutorials/01_naive_generation.py ./Qwen3-0.6B
python nanovllm_pp/tutorials/02_kv_cache.py ./Qwen3-0.6B
python nanovllm_pp/tutorials/03_engine_basic.py ./Qwen3-0.6B

# Benchmarks
python nanovllm_pp/benchmarks/bench_baseline.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_chunked_prefill.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_spec_decode.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_quantization.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_nf4.py ./Qwen3-0.6B
python nanovllm_pp/benchmarks/bench_profiling.py ./Qwen3-0.6B
```

---

## 12. Key Design Decisions

1. **LLM as thin wrapper:** `LLM` inherits `LLMEngine` with no extra logic.
   The public API stays stable while internals can refactor.

2. **BlockManager is independent:** not coupled to scheduler or model runner.
   Testable in isolation with mock sequences.

3. **Sampler is a separate class:** enables plugging in speculative decoding,
   guided decoding, and RL rollouts without modifying model runner.

4. **HF's `past_key_values` for KV cache:** the engine uses HuggingFace's
   built-in cache rather than managing raw KV tensors. Simpler, correct,
   and sufficient for a single-GPU educational engine.

5. **Relative imports throughout:** all internal imports use `from .` or
   `from ..` paths. The package installs and runs from any directory.

6. **`BitsAndBytesConfig` over raw kwargs:** NF4/INT8 loading uses the
   standard `quantization_config` parameter, compatible with all model
   architectures including multi-modal ones.
