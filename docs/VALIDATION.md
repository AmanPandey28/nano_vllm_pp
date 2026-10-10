# Publication validation — October 10, 2026

Checked on the RTX 5050 Laptop GPU, with host CUDA/local socket access, using
the serving environment described in [REPRODUCING.md](REPRODUCING.md).
These checks validate executable workflows; they do not replace the dated
October 6 performance experiments.

| Check | Outcome |
|---|---|
| `pytest -q -o faulthandler_timeout=30` | **113 passed**; CUDA tests executed |
| `ruff check nanovllm_pp tests examples` | Passed |
| `gpu_smoke` | BF16 1024×1024 CUDA matmul passed on SM120 |
| `validate_paged_backend` | Two four-token HF/paged greedy outputs matched; physical prefix reuse demonstrated; all 64 page references released |
| Editable installation in a temporary environment + `examples/generate.py` | Both prompts generated 32 tokens; existing Python environment unchanged |
| Small random serving smoke (4 requests, 32 input / 8 output tokens) | All requests completed; all 64 page references released; detailed report kept locally under `runs/` |
| Actual launcher / localhost HTTP | Health, completion, SSE `[DONE]`, metrics passed; zero queued/running requests and zero used blocks after completion |
| Wheel build | Passed; package contains code/license/metadata, no local document assets |
| Local documentation links | All resolved |
| Public artifact consistency tests | CSV and per-case summaries agree with raw trials; README serving rates agree with artifacts |

The CPU CI workflow runs lint/tests on Python 3.10 and 3.12 using PyTorch
2.10.0 CPU, Transformers 5.8.1 and Accelerate 1.12.0, without downloading a
checkpoint. Its GPU tests intentionally skip; only a GPU-visible
run validates the kernels. Remote CI outcomes are separate from these local
checks and must be inspected on GitHub.

The October 6 artifacts retain their original timestamps, source dirty-state
metadata and measured values. Published local interpreter/checkpoint/dataset
paths are normalized; they are not evidence of a new performance run today.
