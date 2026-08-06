# NanoVLLM++ design and correctness contract

This document describes the executable implementation, including its limits.
It deliberately separates a concept represented in Python from a production
mechanism executed by an accelerator.

## 1. Scope

NanoVLLM++ is a single-process educational inference runtime with two model
backends. The Hugging Face backend is a compatibility oracle. The CUDA Qwen3
backend makes the vLLM data plane concrete: a global fixed-size K/V page pool,
per-request block tables, tensorized slot mappings, Triton cache writes, direct
paged decode attention, and physically shared prefix pages.

Its strongest value is inspectability. It is not production vLLM: paged
prefill uses a readable gather plus PyTorch SDPA, mixed prefill/decode work is
split into two model forwards, and graph capture, preemption/swap, distributed
execution, broad model support, and production kernel tuning are absent.

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
       |-- BlockManager                      page allocation + block tables
       |-- ModelRunner                       HF compatibility path
       |     `-- Hugging Face model          per-request DynamicCache
       `-- PagedModelRunner                  physical Qwen3 CUDA path
             |-- PagedQwen3ForCausalLM       transformer computation
             |-- global [2,L,P,B,Hkv,D] KV   physical cache ownership
             |-- Triton kernels              slot writes + paged decode
             `-- PagedSpecDecodeManager      accepted-prefix commit
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
- `block_table[logical_block]` contains the corresponding reserved physical
  page ID. `BlockManager.slot_mapping()` translates token positions to flat
  page slots for control-plane validation.
- On the HF backend, `_kv_cache` is the per-request DynamicCache.
- On the paged backend, `block_table` is the only request-local K/V address
  state; tensor bytes live in the runner's global pool.

Completion is checked immediately after prefill produces the first output
token. Finishing releases page references and deletes an HF DynamicCache when
present. Zero-reference physical pages remain reusable by hash until their page
ID is selected for fresh allocation, at which point stale hash identity and
physical-validity metadata are cleared.

## 4. KV-cache invariants in both backends

After completed prefill, cache storage contains all prompt tokens and the
sequence contains prompt plus the first generated token. During normal decode:

```text
cache length = sequence token count - 1
cache covers every token before seq.last_token
```

`run_decode()` supplies `seq.last_token`. HF returns a cache extended by that
input. The paged runner writes its K/V to the translated physical slot and then
publishes that slot as computed. Sampling appends a new uncached last token.

Chunked prefill preserves a different intermediate invariant: only the
already-computed prompt prefix is physically valid. HF carries DynamicCache
between chunks. The paged path begins the next chunk at
`num_computed_tokens`, writes through the same block table, and attends to the
logical context gathered from prior pages. The deterministic HF test double
records cache lengths `0, 2, 4` for a five-token prompt split `2 + 2 + 1`.

## 5. Physical pages, block tables, and prefix validity

For element width `b`, decoder layers `L`, K/V heads `Hkv`, and head dimension
`dh`, ordinary dense K/V storage per token is:

```text
2 * L * Hkv * dh * b
```

The factor two is key plus value. For page size `B`, the paged runner allocates
one tensor with shape `[2, L, P, B, Hkv, dh]`. `P` is either the explicit
`kv_cache_num_blocks` override or an estimate from live CUDA headroom. Prompt
admission includes one output position; decode and speculation reserve their
maximum lookahead conservatively.

Device selection is explicit. `device="cuda"` fails when CUDA is hidden rather
than silently benchmarking the CPU; `device="auto"` retains convenient CPU
fallback for functional examples. Automatic CUDA dtype selection uses BF16
when supported and FP16 otherwise. K/V capacity uses that compute/cache dtype,
not the storage width of a quantized parameter.

For token position `p`, logical page `p // B` selects a physical page ID from
the sequence block table, and flat slot `page_id * B + p % B` selects one row
inside the pool. `PagedModelRunner` converts positions, slots, context lengths,
query offsets, and block tables to device tensors. Each attention layer:

1. projects and applies Q/K normalization and RoPE;
2. scatters new K/V to its physical page through a Triton kernel;
3. runs gathered causal SDPA for prefill, or an online-softmax Triton decode
   kernel that reads non-contiguous pages directly through the block table.

The allocator separates token identity from physical validity. Admission knows
all prompt token IDs, but `Block.computed_tokens` advances only after every
model layer has enqueued its writes successfully. A full block hash is
published for prefix reuse only after `computed_tokens == block_size`. All
model work is serialized on the same CUDA stream, so later cache reads are
ordered after those writes without a host-wide synchronize. A future
multi-stream runner would need events before publishing across streams. This
two-phase protocol prevents a request from observing uninitialized K/V.

Hashes are chained from parent hash plus local token IDs, so equal local blocks
do not match under different ancestors. Reuse is full-block only. If a prompt
ends exactly on a page boundary, the final page is deliberately recomputed:
the KV cache stores states, not next-token logits, so at least one query token
must execute to obtain the first output distribution. This also avoids partial
shared-page copy-on-write in the implemented policy.

The HF backend still uses DynamicCache and therefore rejects physical prefix
caching. The same allocator is used for admission metadata, but only the paged
backend makes its addresses part of model execution.

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
prompts could fit. The HF compatibility runner invokes the model per item. The
paged runner flattens all scheduled prefill tokens into one projection/model
batch and all ordinary decode requests into another. A mixed scheduling
iteration therefore uses at most two forwards because prefill needs causal
multi-token attention while decode uses the direct page kernel. Output row
indices scatter sampled tokens back to requests.

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
  the target correction. HF crops overcomputed cache state. Paged execution
  leaves rejected bytes uncommitted; logical lengths hide them and a later
  write overwrites their slots.
- If all `k` match, emit all proposals plus the bonus; the returned cache
  already excludes only that bonus.
- If EOS or a stop-token ID occurs inside the burst, truncate there and commit
  only positions before the terminal token.
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
and TTFT/TPOT/E2E metrics, completions, chat completions, and SSE streaming.

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
seeds, block lifecycle/hash eviction, the physical-validity publication gate,
prefix sharing, exact-boundary recomputation, non-contiguous page scatter and
gather, grouped-query attention, chunk cache carry-over, speculative
rejection/full acceptance/EOS truncation, scheduler lookahead and rotation,
INT4 odd-width packing, storage accounting, profiler math, and JSONL traces.

CUDA-marked tests additionally compile the Triton paged-decode kernel on the
selected GPU, compare it with the PyTorch page-table oracle, execute packed
INT4/INT8 teaching layers, and verify the device manifest. They skip when CUDA
is invisible. The end-to-end validator loads HF and custom Qwen3 paths, prints
greedy parity, exposes the physical tensor shape, and demonstrates prefix reuse.

```bash
ruff check nanovllm_pp tests
pytest -q
python -m nanovllm_pp.benchmarks.gpu_smoke
python -m nanovllm_pp.benchmarks.validate_paged_backend ./Qwen3-0.6B
```

The retained [RTX 5050 validation record](GPU_VALIDATION.md) covers one local
Qwen3-0.6B environment, including SM120 Triton compilation, physical prefix
reuse, paged speculation, and comparison with the HF backend. Broader
qualification remains necessary: long-context stress, server tail latency,
task-level quantized accuracy, thermal control, and workload sweeps.

## 13. Production roadmap

1. Replace gathered SDPA prefill with a fused variable-length paged kernel.
2. Fuse mixed prefill/decode execution where the accelerator backend supports it.
3. Add preemption/eviction and swap or recompute policy under memory pressure.
4. Add partial-page prefix copy-on-write and a bounded cache eviction policy.
5. Add compiled shape buckets, graph-break analysis, and CUDA/accelerator graphs.
6. Replace naive quantization with calibrated formats and fused kernels.
7. Add draft-model stochastic speculation with lossless rejection sampling.
8. Add tensor/pipeline/expert parallelism and distributed scheduling.
9. Establish correctness, accuracy, performance, and SLO regression gates.
