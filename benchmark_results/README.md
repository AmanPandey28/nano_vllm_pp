# Measurement artifacts

These files are dated observations, not expected outputs or CI speed thresholds.
Timings and token/quality measurements are unchanged. Local interpreter/model/
dataset paths in published metadata are normalized to portable paths; version,
checkpoint hashes, measurement timestamps and recorded dirty-source state are
retained. No private documents or model weights belong in this directory.

## October 6, 2026 — repeated controlled experiments

Start with [runtime_evidence/summary.csv](runtime_evidence/summary.csv), then
[methodology](../docs/RUNTIME_EVIDENCE.md).

| File(s) in `runtime_evidence/` | What they contain |
|---|---|
| [results.json](runtime_evidence/results.json) | All 12 serving cases, three trials each, exact-output comparisons |
| `mixed_batch{1,4,8}.json` | Same-runtime batching width sweep |
| `mixed_batch8_rate4.json` | Seeded Poisson arrivals rather than a burst |
| `prefix{64,256,512}_{off,warm}.json` | Prefix-length and physical reuse ablations |
| `prefix256_cold_start.json` | Cold prefix index after kernel warmup |
| `mixed_pressure72_batch2.json` | Near-full pool and final reference reclamation |
| [quantized_loading.json](runtime_evidence/quantized_loading.json) | Separate HF loading, memory, timing, tokens and small fidelity probe |

Per-case JSON includes prompt text/IDs, generated IDs, arrival/send/token/end
timestamps, scheduler-iteration queue/budget/page samples and final pool state.
The aggregate additionally records cross-case completion matches. The large
JSON is intentional audit evidence; CSV is the reviewer-friendly entry point.

The manifest records base revision `fe112115` plus dirty source paths from the
measurement worktree; it is not a claim that pristine `fe112115` produced the
numbers. The runtime improvements and harnesses are published together in the
subsequent evidence commit. Later packaging/documentation/CLI changes do not
turn the October 6 timings into measurements of a later clean commit.

## August 24, 2026 — historical workload sweep

[rtx5050_2026-08-24/](rtx5050_2026-08-24/) contains 12 JSON reports: fixed-shape
random workloads, concurrency, HF reference, prefix on/off and ShareGPT arrival
rates. These are single warmed trials with different settings and software.
Use [STANDARD_BENCHMARKS.md](../docs/STANDARD_BENCHMARKS.md) rather than comparing
their peak directly with the October mixed-workload medians.

## Re-run without changing evidence

Follow [REPRODUCING.md](../docs/REPRODUCING.md). New experiment output defaults
to ignored `runs/`. Publish a new dated directory only after checking the input
hashes, protocol, output correctness and complete environment metadata.
