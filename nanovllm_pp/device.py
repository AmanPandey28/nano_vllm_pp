"""Device and dtype selection shared by the engine, tutorials, and benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a CPU/CUDA device and fail loudly for unavailable CUDA.

    ``auto`` is convenient for examples. Performance runs should pass
    ``cuda`` explicitly so an isolated environment cannot silently benchmark
    the CPU when the host GPU is hidden.
    """
    if not isinstance(requested, str):
        raise TypeError("device must be a string")
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device {requested!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA devices are supported")
    if device.type == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see a CUDA device. Run the "
            "process with access to /dev/nvidia* (outside a restricted sandbox) "
            "and verify it first with `python -m nanovllm_pp.benchmarks.gpu_smoke`."
        )
    index = 0 if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is out of range for "
            f"{torch.cuda.device_count()} visible device(s)"
        )
    return torch.device(f"cuda:{index}")


def resolve_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    """Choose an inference dtype that the selected device can execute."""
    if not isinstance(requested, str):
        raise TypeError("dtype must be a string")
    requested = requested.strip().lower()
    if requested == "auto":
        if device.type == "cpu":
            return torch.float32
        with torch.cuda.device(device):
            supports_bfloat16 = torch.cuda.is_bf16_supported()
        return torch.bfloat16 if supports_bfloat16 else torch.float16
    if requested not in _DTYPES:
        choices = ", ".join(["auto", *_DTYPES])
        raise ValueError(f"dtype must be one of: {choices}")
    if device.type == "cpu" and requested == "float16":
        raise ValueError("float16 inference is not supported by this CPU path")
    return _DTYPES[requested]


def model_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Use the non-deprecated Transformers dtype spelling when available."""
    import transformers

    version = transformers.__version__.split(".")
    try:
        major, minor = int(version[0]), int(version[1])
    except (IndexError, ValueError):
        return {"torch_dtype": dtype}
    return {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": dtype}


def synchronize(device: str | torch.device) -> None:
    """Wait for queued CUDA work; a no-op for CPU."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def timed_call(
    function: Callable[..., Any],
    *args: Any,
    device: str | torch.device,
    **kwargs: Any,
) -> tuple[Any, float]:
    """Measure a call without undercounting asynchronous CUDA execution."""
    synchronize(device)
    start = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize(device)
    return result, time.perf_counter() - start


def cuda_device_info(device: str | torch.device = "cuda:0") -> dict[str, Any]:
    """Return stable fields useful in benchmark environment manifests."""
    resolved = resolve_device(str(device))
    if resolved.type != "cuda":
        raise ValueError("cuda_device_info requires a CUDA device")
    props = torch.cuda.get_device_properties(resolved)
    with torch.cuda.device(resolved):
        supports_bfloat16 = torch.cuda.is_bf16_supported()
    return {
        "device": str(resolved),
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_mib": round(props.total_memory / (1024**2)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "bfloat16_supported": supports_bfloat16,
    }
