# RTX 5050 serving-workload benchmarks

The October 6 controlled same-runner batching, warm/cold prefix, and bounded
KV-pressure experiments are in [RUNTIME_EVIDENCE.md](RUNTIME_EVIDENCE.md).
This document retains the earlier standard-workload sweep and its original
software/measurement context.

This report records the standard-workload sweep run on August 24, 2026. The
machine, checkpoint, and software stack match [GPU_VALIDATION.md](GPU_VALIDATION.md).
Raw JSON results are retained under
[`benchmark_results/rtx5050_2026-08-24`](../benchmark_results/rtx5050_2026-08-24/).

These are engine-level laptop measurements, not production capacity claims.
They are useful for identifying shape, concurrency, arrival, and prefix-cache
effects in this implementation.

## Method

- Model: local Qwen3-0.6B BF16 checkpoint.
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, compute capability 12.0.
- Paged cache: 512 pages, 16 tokens/page, 8,192-token total capacity;
  `[2,28,512,16,8,128]` = 939,524,096 bytes.
- Scheduler: chunked prefill enabled, decode priority enabled, 2,048-token
  iteration budget.
- Generation: greedy, EOS ignored so every request emits its declared length.
- Timing: model load and warm-up excluded; CUDA synchronized at every engine
  step because token arrival times are observed at the public step boundary.
- Triton warm-up: one representative request per power-of-two context bucket,
  preventing JIT compilation from appearing as measured ITL.
- Closed concurrency: at most `--max-concurrency` clients are outstanding.
  Requests waiting behind this client-side semaphore contribute to the
  separately reported `client_wait`, not TTFT. Benchmark duration runs from
  the first send to the final completion.
- Arrival: finite rates use seeded exponential inter-arrival times; `inf`
  makes all clients immediately eligible.

The workload names and primary metrics follow current vLLM benchmark
conventions: fixed-shape random data, ShareGPT, and prefix repetition; request,
input-token, output-token, and total-token throughput; TTFT, TPOT, ITL, and E2E.
See the official [vLLM benchmark CLI documentation](https://docs.vllm.ai/en/latest/cli/bench/serve/).

Each row below is one warmed synchronized run. Laptop clocks and power were not
locked. The latency percentiles from 8--16 requests are descriptive and should
not be treated as statistically stable production percentiles.

## Fixed-shape phase sweep

| Shape and concurrency | Requests | Request/s | Input tok/s | Output tok/s | Total tok/s | TTFT p50 | TPOT p50 | ITL p50/p95 | Max pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decode-heavy 32/128, C=8 | 16 | 4.068 | 130.2 | 520.7 | 650.9 | 45.5 ms | 15.1 ms | 15.0/15.5 ms | 80 |
| Balanced 256/64, C=8 | 16 | 6.614 | 1,693.2 | 423.3 | 2,116.5 | 219.4 ms | 15.7 ms | 15.6/16.5 ms | 160 |
| Prefill-heavy 1024/16, C=4 | 8 | 5.574 | 5,707.3 | 89.2 | 5,796.5 | 463.1 ms | 16.8 ms | 15.8/20.3 ms | 260 |

The decode-heavy row exposes steady generation rate. The prefill-heavy row has
high input-token throughput but substantially higher TTFT. `total tok/s` must
therefore not be substituted for output throughput or user-visible latency.

## Balanced-shape concurrency sweep

All rows use 16 requests with 256 input and 64 output tokens.

| Max concurrency | Request/s | Output tok/s | Total tok/s | TTFT p50 | TPOT p50 | E2E p50 | Max pages |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.116 | 71.4 | 357.0 | 20.7 ms | 14.0 ms | 902.3 ms | 20 |
| 4 | 3.968 | 254.0 | 1,269.8 | 106.2 ms | 14.2 ms | 1,007.7 ms | 80 |
| 8 | 6.614 | 423.3 | 2,116.5 | 219.4 ms | 15.7 ms | 1,208.2 ms | 160 |

Batching raises aggregate throughput while increasing TTFT and per-request E2E
latency. TPOT remains comparatively stable. This is why serving optimization
must state whether it targets throughput, first-token responsiveness, or
streaming cadence.

## HF reference versus paged execution

The balanced 256/64, concurrency-8 workload was repeated through the HF
reference runner.

| Backend | Request/s | Output tok/s | TTFT p50 | TPOT p50 | ITL p50 | E2E p50 | Resident/peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| HF reference | 1.157 | 74.0 | 300.9 ms | 105.0 ms | 104.0 ms | 6,913.6 ms | 1.14/1.43 GiB |
| Physical paged | 6.614 | 423.3 | 219.4 ms | 15.7 ms | 15.6 ms | 1,208.2 ms | 2.02/2.07 GiB |

This is a whole-path comparison. The HF runner dispatches scheduled requests
serially; the paged runner batches same-mode requests, selects only needed
logit rows, and uses direct Triton paged decode. The 5.72x output-throughput
ratio does **not** isolate page-table addressing as the cause. The paged path
also reserves a fixed 896 MiB physical K/V pool up front.

Production vLLM was not installed in this environment, so no custom-versus-vLLM
performance claim is made.

## Physical prefix reuse

Eight requests each contain 288 prompt tokens: a 256-token shared prefix plus
a 32-token unique suffix. A separate cache-populating request is run before the
measurement in both cases and excluded from timing.

| Prefix cache | Cached/input tokens | Hit fraction | Max pages | Output tok/s | TTFT p50 | E2E p50 |
|---|---:|---:|---:|---:|---:|---:|
| Disabled | 0/2,304 | 0.000 | 160 | 362.7 | 177.3 ms | 690.2 ms |
| Enabled | 2,048/2,304 | 0.889 | 48 | 448.2 | 52.3 ms | 569.3 ms |

With caching enabled, every measured request reused 256 physically valid K/V
tokens. Page demand fell 70%, p50 TTFT fell 70.5%, and output throughput rose
23.6%. These gains apply to a warm shared-prefix workload; cold-cache behavior
still includes the first prefix computation.

## ShareGPT

The dataset is the current vLLM-documented ShareGPT file. Local SHA-256:
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.
Records are shuffled with seed 17, bounded during tokenization, and filtered to
fit `max_model_len=2048`.

The variable-output run used 16 requests. Input length was 5--1,111 tokens
(mean 164.81, median 21); output length was 7--128 tokens (mean 109.75,
median 128).

| Request/s | Input tok/s | Output tok/s | Total tok/s | TTFT p50/p95/p99 | TPOT p50/p95/p99 | ITL p50/p95/p99 | E2E p50/p95/p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.586 | 591.0 | 393.6 | 984.6 | 53.0/159.6/179.1 ms | 17.5/18.1/18.4 ms | 15.7/19.6/61.6 ms | 2,230.3/2,404.3/2,461.1 ms |

## ShareGPT arrival-rate sweep

The same 12 sampled prompts and fixed 32-token outputs are used in every row.
Input length is 5--1,111 tokens (mean 139.08, median 21).

| Offered requests/s | Observed max concurrency | Achieved request/s | Output tok/s | TTFT p50/p95/p99 | TPOT p50/p95/p99 | ITL p50/p95/p99 | E2E p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.982 | 31.4 | 35.3/123.9/206.6 ms | 15.3/24.2/24.3 ms | 15.2/19.9/56.3 ms | 509.3 ms |
| 4 | 5 | 3.506 | 112.2 | 35.0/123.0/179.0 ms | 15.6/20.2/20.3 ms | 14.8/25.1/65.9 ms | 522.5 ms |
| infinity | 8 | 9.966 | 318.9 | 60.8/114.0/165.1 ms | 15.8/18.4/18.4 ms | 15.6/16.3/37.3 ms | 550.4 ms |

Higher offered load improves batching and aggregate throughput but raises the
median first-token delay. Long-prompt outliers dominate the small-sample tail.
A production study should use hundreds of requests, repeated seeds, locked
power, confidence intervals, and an SLO-goodput curve.

## Reproducing a run

```bash
python -m nanovllm_pp.benchmarks.bench_serving ./Qwen3-0.6B \
  --dataset-name random \
  --num-prompts 16 \
  --random-input-len 256 \
  --random-output-len 64 \
  --max-concurrency 8 \
  --request-rate inf \
  --kv-cache-num-blocks 512 \
  --max-model-len 512 \
  --save-result runs/result.json
```

Use `--dataset-name sharegpt --dataset-path DATA.json` for ShareGPT, or
`--dataset-name prefix_repetition` with `--enable-prefix-caching` and
`--no-enable-prefix-caching` for the controlled prefix comparison. Run
`python -m nanovllm_pp.benchmarks.bench_serving --help` for every flag.

## Important limitations

- The harness drives the engine directly. It includes tokenization after the
  send timestamp but excludes HTTP, SSE, network, and client parsing overhead.
- Measurements are single runs after warm-up; there are no confidence
  intervals and laptop power/thermal state was not locked.
- Greedy lengths are forced with `ignore_eos=True`; this controls work but is
  not an output-quality evaluation.
- Input tok/s counts prompt tokens completed during the end-to-end serving
  interval. It is not isolated prefill-kernel throughput.
- Prefix results describe a warm cache and do not include warm-up cost.
- The current gathered prefill path is correct but not a fused paged-prefill
  kernel. Decode is the direct page-table Triton path.
- Small-sample p95/p99 values are retained for schema consistency, not strong
  tail-latency inference.
