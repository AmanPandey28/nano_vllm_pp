# NanoVLLM++ design and correctness contract

This document describes the executable implementation, including its limits.
It deliberately separates a concept represented in Python from a production
mechanism executed by an accelerator.

## 1. Scope

NanoVLLM++ is a single-process educational inference runtime around a
Transformers causal language model. Its strongest value is inspectability:
each request has explicit state, the scheduler has explicit queues and budgets,
and prefill, decode, speculation, and completion are separate operations.

The project does **not** implement a physical paged-attention cache, a fused
multi-request model batch, graph compilation, custom attention kernels, or
distributed execution. Those are production roadmap items, not hidden
assumptions.

## 2. Ownership and call graph

```text
caller / FastAPI
       |
       v
LLM -> LLMEngine
       |-- tokenizer                         text <-> token IDs
       |-- Sequence registry                 request state and results
       |-- Scheduler                         waiting/running policy
       |     `-- MixedBatchScheduler         chunking and token budget
       |-- BlockManager                      logical capacity accounting
       `-- ModelRunner                       HF forward + sample
              |-- Hugging Face model         actual compute and KV tensors
              `-- SpecDecodeManager          optional exact greedy path
```

`LLM` is a deliberately empty facade over `LLMEngine`. `LLMEngine.step()` is
the central loop: schedule work, run it, detect stop strings, postprocess state,
record metrics/traces, and return newly finished sequences.

## 3. Request state machine

A prompt becomes a `Sequence` with immutable `prompt_len` and a growing
`token_ids` list.

```text
WAITING --prefill admission--> RUNNING --EOS/stop/length--> FINISHED
```

Important counters:

- `num_computed_tokens` counts prompt tokens processed by prefill.
- `num_generated_tokens` counts tokens appended after the prompt.
- `generated_token_ids == token_ids[prompt_len:]`.
- `block_table` contains reserved logical page IDs.
- `_kv_cache` is the actual per-request Hugging Face cache while active.

Completion is checked immediately after the prefill produces the first output
token. This matters for `max_tokens=1` and for an EOS first token. Finishing a
request releases logical blocks and deletes its K/V cache. Completed token
state remains in the engine registry until the output is consumed; synchronous
`generate()` and the server call `forget_request()` automatically.

## 4. Actual KV-cache invariant

After a completed prefill, the Hugging Face cache contains all prompt tokens
and the sequence contains prompt plus the first generated token. During normal
decode the invariant is:

```text
cache length = sequence token count - 1
cache covers every token before seq.last_token
```

`run_decode()` supplies `seq.last_token` with the old cache. The returned cache
therefore includes that input token, and the sampled output becomes the new
uncached last token.

Chunked prefill preserves a different intermediate invariant: the cache holds
exactly the already-computed prompt prefix. Every later chunk passes that cache
as `past_key_values`; omitting it would reset attention context and position
semantics. The deterministic test double records cache lengths `0, 2, 4` for a
five-token prompt split as `2 + 2 + 1`.

## 5. Logical block accounting—not PagedAttention

For element width `b`, decoder layers `L`, K/V heads `Hkv`, and head dimension
`dh`, ordinary dense K/V storage per token is:

```text
2 * L * Hkv * dh * b
```

The factor two is key plus value. `LLMEngine` estimates how many fixed-size
logical blocks fit under current CUDA free memory and the configured
utilization target, with extra runtime headroom. The scheduler reserves these
blocks before admission and decode. Speculation reserves its maximum upcoming
positions conservatively.

Device selection is explicit. `device="cuda"` fails when CUDA is hidden rather
than silently benchmarking the CPU; `device="auto"` retains convenient CPU
fallback for functional examples. Automatic CUDA dtype selection uses BF16
when supported and FP16 otherwise. K/V capacity uses that compute/cache dtype,
not the storage width of a quantized parameter.

`BlockManager` stores IDs, refcounts, hashes, and token metadata only. No layer
K/V tensor is indexed through `block_table`; actual tensors grow inside the HF
cache. Therefore:

- logical capacity and lifecycle behavior are implemented;
- physical non-contiguous paging and an attention slot mapping are not;
- prefix hashes are testable as a library concept, but engine-level prefix
  caching is rejected because metadata reuse without tensor reuse is false.

A production implementation would preallocate per-layer K/V pools, assign
physical blocks, create slot/block mappings for each batch, write new K/V into
those slots, and make attention gather through the block table.

## 6. Scheduler and chunked prefill

Without chunking, waiting requests are admitted for full prefill under
`max_num_batched_tokens` and `max_num_seqs`; decode begins when the waiting
queue is empty.

With chunking, `MixedBatchScheduler` builds one iteration:

1. Schedule running decode requests first when `decode_priority=True`.
2. Charge one token for normal decode or conservative lookahead for speculation.
3. Spend remaining budget on at most one prefill chunk per waiting request.
4. Rotate a partially processed long prompt to the back of the deque.
5. Optionally reverse the phase order when decode priority is disabled.

Rotation avoids a long prompt monopolizing the head of the FIFO while short
prompts could fit. The policy is iteration-level continuous scheduling, but
`ModelRunner.run()` loops over items and invokes the HF model separately. A
production batch would flatten tokens, build position/slot metadata, issue one
batched graph, and scatter outputs back to sequences.

If unfinished work exists but no item can be scheduled, `LLMEngine.step()`
raises with queue/block state instead of spinning forever.

## 7. Sampling and termination

`Sampler` clones model logits before in-place filtering, then applies:

1. temperature scaling;
2. top-k masking;
3. top-p nucleus masking;
4. min-p masking relative to the maximum probability;
5. argmax for temperature zero or multinomial sampling otherwise.

A per-sequence `torch.Generator` makes seeded stochastic sampling reproducible
without resetting every step. Optional selected-token and top-N raw model
log-probabilities are retained. Parameters are validated at construction.

Finish conditions are stop string, stop-token ID, model EOS (unless ignored),
or `max_tokens`. String detection decodes the generated suffix after each
scheduled step and trims the returned text at the first match.

## 8. Exact greedy n-gram speculative decoding

The n-gram proposer finds a suffix that occurred earlier and proposes the
tokens that followed that occurrence. It has no draft model and no guarantee
that a useful match exists.

Suppose the normal cache excludes current last token `x`, and the proposer
returns `[d1, d2, ..., dk]`. One target call receives:

```text
[x, d1, d2, ..., dk] + old cache
```

Logit row zero predicts `d1`; row one predicts `d2`; and row `k` predicts a
bonus token. Verification accepts the longest proposal prefix matching target
argmax.

- On the first mismatch after `j` accepted tokens, emit those `j` tokens plus
  the target correction, then crop overcomputed cache state to exclude the
  correction.
- If all `k` match, emit all proposals plus the bonus; the returned cache
  already excludes only that bonus.
- If EOS or a stop-token ID occurs inside the burst, truncate there and crop to
  restore the cache invariant.
- Stop strings and stochastic sampling use the ordinary one-token path. Exact
  stochastic speculative decoding would require rejection sampling and a
  draft probability distribution, which this n-gram proposer does not have.

Acceptance rate means accepted draft tokens divided by proposed draft tokens;
the target correction and bonus are useful emitted tokens but are not accepted
draft tokens.

## 9. Quantization

The custom teaching path performs symmetric per-output-channel quantization:

```text
scale[c] = max(abs(weight[c, :])) / qmax
q[c, :]  = round(weight[c, :] / scale[c])
```

INT8 stores one signed byte per weight. INT4 shifts signed values into nibbles
and packs two input columns per byte; odd input widths receive one zero-valued
padding nibble and are trimmed during dequantization. Wrapper modules register
quantized weights and scales as buffers, preserving actual storage accounting.

Both wrappers reconstruct a floating weight on every call before matrix
multiplication. They demonstrate accuracy/storage trade-offs, not an inference
speedup. A production path needs a fused dequantize-matmul kernel, suitable
grouping/calibration, hardware-supported formats, and model-level accuracy
evaluation.

The separate `load_in_4bit`/`load_in_8bit` path delegates to Transformers and
bitsandbytes on CUDA. Quantized models are placed during loading and are not
subsequently passed through `.to()`. The two flags are mutually exclusive and
CPU use is rejected explicitly.

## 10. Serving and concurrency

`create_app()` returns a configured FastAPI instance; it does not put engine
state in a mutable module global. Endpoints include health, logical block/queue
metrics, completions, chat completions, and SSE streaming.

The engine is not thread-safe, so one `asyncio.Lock` protects admission,
stepping, snapshots, cleanup, and metrics reads. Model work runs in a worker
thread so the event loop remains responsive. Each coroutine yields between
steps, allowing later requests into later scheduler iterations. This is safe
cooperative access, not parallel model execution.

Streaming takes a decoded-text snapshot after each engine step and sends the
delta, then a final finish-reason chunk and `[DONE]`. Chat uses the tokenizer's
chat template when available and a simple tagged fallback otherwise.

## 11. Observability

- `EngineProfiler` measures schedule, prefill, decode, mixed, and postprocess
  wall times per engine step.
- `RequestTracker` reports nearest-rank p50/p95/p99 TTFT, E2E, and TPOT. For N
  generated tokens, TPOT divides post-first-token time by `N - 1` intervals.
- `EngineTrace` records enqueue, schedule, finish, and step-complete events and
  can write JSON Lines for offline inspection.

These host-level timers are a first diagnostic layer. Kernel attribution,
device utilization, bandwidth, graph breaks, and host/device gaps require
PyTorch Profiler, accelerator tools, or Nsight on a real deployment.

## 12. Verification

The deterministic CPU suite covers configuration validation, sampling and
seeds, block lifecycle/hash eviction, chunk cache carry-over, max-token
completion, speculative rejection/full acceptance/EOS truncation, scheduler
lookahead and rotation, INT4 odd-width packing, storage accounting, profiler
math, and JSONL traces.

CUDA-marked tests additionally execute the packed INT4 and INT8 teaching
layers on the selected GPU and verify the device manifest. They skip when CUDA
is not visible, while `python -m nanovllm_pp.benchmarks.gpu_smoke` is the
fail-fast host/device check used before performance runs.

```bash
ruff check nanovllm_pp tests
pytest -q
python -m nanovllm_pp.benchmarks.gpu_smoke
```

The retained [RTX 5050 validation record](GPU_VALIDATION.md) covers one local
Qwen3-0.6B environment: CUDA smoke, all 37 tests, greedy cache/engine parity,
chunk parity, speculative parity, quantized storage/timing, and PyTorch CUDA
profiling. Broader qualification remains necessary: stochastic distribution
tests, long-context stress, multiple cache implementations and Transformers
versions, server load/tail latency, task-level quantized accuracy, thermal/power
control, and workload sweeps.

## 13. Production roadmap

1. Replace per-sequence dynamic caches with a preallocated physical K/V pool
   consumed by paged attention.
2. Build real heterogeneous tensor batches and output scatter metadata.
3. Add preemption/eviction and swap or recompute policy under memory pressure.
4. Implement physical prefix-cache reuse with safe invalidation.
5. Add compiled shape buckets, graph-break analysis, and CUDA/accelerator graphs.
6. Replace naive quantization with calibrated formats and fused kernels.
7. Add draft-model stochastic speculation with lossless rejection sampling.
8. Add tensor/pipeline/expert parallelism and distributed scheduling.
9. Establish correctness, accuracy, performance, and SLO regression gates.
