# Documentation

Start with the root [README](../README.md) for scope, results and quick start.

| Document | Use it for |
|---|---|
| [DESIGN.md](DESIGN.md) | Ownership, request state, allocator invariants, page-table execution, speculation and limitations |
| [USAGE.md](USAGE.md) | Python API, server launcher, HTTP fields, cleanup and metric definitions |
| [REPRODUCING.md](REPRODUCING.md) | Isolated environments, checkpoint checksums, tests, benchmark and dataset commands |
| [VALIDATION.md](VALIDATION.md) | October 10 test, GPU, install, server and package checks |
| [RUNTIME_EVIDENCE.md](RUNTIME_EVIDENCE.md) | October 6 repeated same-runner batching, prefix and KV-pressure evidence |
| [QUANTIZATION_SCOPE.md](QUANTIZATION_SCOPE.md) | Loading versus storage versus compute; measured memory/latency/fidelity tradeoffs |
| [STANDARD_BENCHMARKS.md](STANDARD_BENCHMARKS.md) | August 24 historical shape, concurrency, ShareGPT and arrival-rate sweep |
| [GPU_VALIDATION.md](GPU_VALIDATION.md) | Historical GPU bring-up, model parity and device-access checks |
| [REFERENCES.md](REFERENCES.md) | Design lineage and primary external references |

The [artifact index](../benchmark_results/README.md) maps claims to raw JSON/CSV.
Historical measurements retain their dates and settings; they are not replaced
or silently treated as results from the latest checkout.
