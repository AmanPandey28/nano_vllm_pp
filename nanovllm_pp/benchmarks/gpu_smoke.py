"""Fail-fast CUDA visibility and compute smoke test."""

import json

import torch

from nanovllm_pp.device import cuda_device_info, resolve_device, synchronize


def main():
    device = resolve_device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    left = torch.randn((1024, 1024), device=device, dtype=dtype)
    right = torch.randn((1024, 1024), device=device, dtype=dtype)
    output = left @ right
    synchronize(device)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("CUDA matmul produced non-finite values")
    report = cuda_device_info(device)
    report.update(
        {
            "smoke_dtype": str(dtype).removeprefix("torch."),
            "matmul_shape": list(output.shape),
            "allocated_mib": round(torch.cuda.memory_allocated(device) / (1024**2), 1),
            "status": "ok",
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
