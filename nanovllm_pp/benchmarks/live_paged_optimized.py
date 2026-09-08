"""Run the paged backend with chunked prefill and prefix caching enabled."""

import importlib
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_common = importlib.import_module("nanovllm_pp.benchmarks.live_demo_common")


if __name__ == "__main__":
    _common.run_demo(_common.PAGED_OPTIMIZED)
