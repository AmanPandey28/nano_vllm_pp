# Controlled runtime evidence

Measured October 6, 2026 on Qwen3-0.6B / NVIDIA RTX 5050 Laptop GPU (SM120,
7,708 MiB visible memory). The serving experiment used Python 3.11.14,
PyTorch 2.10.0+cu130, Transformers 5.8.1, and Triton 3.6.0. Driver version was
595.91.07. Clocks/power were not locked; three repetitions establish a small
sample of this workload, not a broad statistical or hardware-independent claim.

The model SHA-256, versions, GPU-state sample, command, and code revision/dirty
state are in every JSON manifest. Invocation form is `python -m ...`; local
interpreter/checkpoint paths are normalized for publication, and unrelated
local-document paths are excluded from git-status metadata. No timing, token,
or correctness measurements were changed. The measurement source snapshot is
a dirty worktree based on `fe112115`, with the runtime improvements now
published in the repository; it is not pristine `fe112115`.

## Reproduce

```bash
# Use the serving environment from REPRODUCING.md.
python -m nanovllm_pp.benchmarks.gpu_smoke
python -m nanovllm_pp.benchmarks.bench_runtime_evidence \
  ./Qwen3-0.6B \
  --output-dir runs/runtime_evidence \
  --repeats 3 --num-prompts 12 --seed 17
```

See [REPRODUCING.md](REPRODUCING.md) for dependencies and checkpoint checksums.
Use a terminal/process with `/dev/nvidia*` access. The command writes per-case
JSON, a full `results.json`, and an aggregate `summary.csv` under `runs/`.
The recorded [results.json](../benchmark_results/runtime_evidence/results.json)
and [summary.csv](../benchmark_results/runtime_evidence/summary.csv) stay unchanged.
Each trial records prompt IDs, generated IDs, arrival/send/token/completion
timestamps, per-iteration budget/queue/page samples, memory, and final pool
state. Large JSON files are intentional raw evidence; CSV is the compact entry
point. Warmup and model loading are outside measured serving time.

## Batching versus the same runner with scheduling width one

Twelve fixed synthetic requests use prompt lengths 32/256/512 and output
lengths 16/32, with seed 17 and greedy decoding forced to the requested length
(`ignore_eos=True`). This is a seeded engineering workload rather than a
natural-language quality dataset. The same custom paged Qwen3 model, BF16
weights, tokenizer, 16-token pages, 512-page pool, 1,024-token budget, and
128-token chunk limit are used throughout. Prefix caching and speculation
are disabled. Only scheduling width / arrival rate vary.

All twelve requests pass the client gate together in the burst cases.
`max_num_seqs=1` permits one work item/model request per iteration; it does not
force every request to finish before another request holds partial prompt KV.
It is therefore precisely a **batching-disabled, single-sequence-per-iteration
baseline**, not a different backend or an HTTP client submitting one-at-a-time.

Medians across three warmed trials:

| Scheduling width / arrival | Output tok/s | TTFT p50 / p95 ms | TPOT p95 ms | E2E p95 ms |
|---|---:|---:|---:|---:|
| 1, all at once | 57.969 | 2,648.1 / 4,867.5 | 17.538 | 5,197.4 |
| 4, all at once | 155.522 | 730.6 / 1,475.9 | 21.459 | 1,890.3 |
| 8, all at once | 222.731 | 212.9 / 776.9 | 23.782 | 1,305.1 |
| 8, seeded Poisson 4 requests/s | 90.094 | 77.4 / 158.2 | 21.361 | 756.2 |

Width eight delivers `222.731 / 57.969 = 3.842×` output throughput. Batched
projections use the GPU more efficiently and amortize request-level dispatch.
The saturation burst raises width-eight p95 TTFT from 158.2 to 776.9 ms versus
the arrival-paced case; slower per-request TPOT at larger scheduling widths
shows a second throughput/latency tradeoff. The arrival-paced throughput also
includes idle time, as a serving metric should.

Throughput samples, width one: `[57.969, 59.510, 54.221]`; width eight:
`[215.168, 223.416, 222.731]`. Eleven of twelve greedy completions matched the
single-sequence baseline at widths four and eight in each trial. Batched
floating-point reduction order can change close argmax decisions; the result
does not claim bitwise batch invariance. Tensor-parity tests use explicit
tolerances, including `atol=rtol=0.02` for BF16 paged attention and `0.003` for
FP16 adversarial cases. These are functional checks, not corpus accuracy.

## Prefix reuse, cold start, and prefix-length sweep

Each workload has 12 unique suffixes of 32 tokens, 16 forced output tokens,
scheduling/client concurrency four, and the same settings with only prefix
caching changed. A separate cache-populating request is outside timing for
warm runs. Before every trial, the hash index is cleared; the warmup seeds the
prefix again. The cold-start case clears the index *after* kernel warmup, so
JIT/setup is still excluded but the first request wave computes its prefix.

| Shared prefix / cache state | Reused tokens | Prompt-token hit fraction | TTFT p50 / p95 ms | Output tok/s |
|---|---:|---:|---:|---:|
| 64 / off | 0 | 0 | 34.725 / 36.191 | 235.379 |
| 64 / warm | 768 | 66.67% | 40.049 / 43.301 | 216.710 |
| 256 / off | 0 | 0 | 139.124 / 151.483 | 170.994 |
| 256 / warm | 3,072 | 88.89% | 34.153 / 35.381 | 235.012 |
| 256 / cold start | 2,048 | 59.26% | 34.635 / 137.018 | 207.329 |
| 512 / off | 0 | 0 | 261.184 / 264.416 | 128.175 |
| 512 / warm | 6,144 | 94.12% | 35.108 / 37.209 | 235.875 |

The 256-token warm case saves exactly `12*256=3,072` prompt forwards, leaving
384 uncached prompt tokens rather than 3,456. Its median TTFT falls 75.45%.
All 12 generated token sequences match the cache-off run for all three prefix
lengths in every trial. The final prompt block remains a query-producing block;
partial pages are never shared.

At 64 tokens the overhead/noise overwhelms any stable benefit: warm throughput
samples are `[118.234, 216.710, 240.638]`. This negative result stays in the
evidence. Cold start has a long first-wave TTFT, while later waves reuse pages
released by earlier requests; calling its median a fully warm result would be
misleading. Physical pages with live references are never overwritten.

## KV pressure and failure behavior

The same mixed workload also completes with a 72-page pool and scheduling width
two. Peak live usage is **70/72 pages (97.22%)**, throughput 93.932 output tok/s,
and p95 TTFT 2,773.0 ms. All request references are reclaimed at the end.
This case demonstrates bounded capacity and reclamation; pool size and width
both differ from the width-eight run, so it does not isolate a pool-size-only
performance effect. Every trial asserts an empty live-reference set.

There is no production preemption or swap. A genuine no-progress pool shortage
raises a diagnostic error. `abort_request()` can release a request; the caller
can also reduce admitted work or enlarge the pool. Unit tests demonstrate that
failed multi-page reservations are atomic and that blocked waiting prompts
allow running decode to finish and free pages.

## Correctness evidence and claim locations

The final host-visible suite passed **104 tests**; `ruff check nanovllm_pp
tests` passed. `tests/test_kv_invariants.py` includes 18 boundary cases and five
300-iteration seeded audits (1,500 checked iterations total), plus explicit
sharing, eviction, stale identity, and atomic failure checks. Scheduler tests
cover token-budget fairness, blocked admission, output boundary reservation,
and cancellation of a shared partial prefill. `tests/test_request_cleanup.py`
checks model failures, idempotent cancellation, transactional HTTP submission,
and unfinished stream closure. `tests/test_paged_kv.py` compares noncontiguous
FP16/BF16 page writes and decode with an oracle and poisons unused slots with NaN.

```bash
pytest -q -o faulthandler_timeout=30
ruff check nanovllm_pp tests
```

Run asynchronous streaming tests with host/local socket access: the restricted
automation sandbox can stall asyncio thread-completion wakeups. CPU-only
environments skip CUDA tests; that is not GPU validation.

| Claim | Evidence / implementation |
|---|---|
| Same-runtime 3.84× throughput | `results.json`: `mixed_batch1` versus `mixed_batch8`; `bench_runtime_evidence.py` |
| 256-token prefix TTFT/reuse | `prefix256_off.json` / `prefix256_warm.json`; every warm trial stores exact-output match count |
| Fixed pages / references / parent hashes | `engine/block_manager.py`, `paged/runner.py`, `test_kv_invariants.py` |
| Quantized loading / storage prototypes | `quantized_loading.json`, `bench_loading_evidence.py`, `QUANTIZATION_SCOPE.md` |
| API lifecycle and telemetry | `server/app.py`, `observability/profiler.py`, `test_request_cleanup.py` |

No comparison with production vLLM is used as a result. Broad accuracy,
goodput/SLO attainment, graph capture, distributed inference, LRU eviction,
and fused paged prefill remain outside this runtime's measured scope.
