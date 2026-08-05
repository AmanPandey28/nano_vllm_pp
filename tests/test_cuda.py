import pytest
import torch

from nanovllm_pp.device import cuda_device_info, resolve_device
from nanovllm_pp.quantization.int8_weight_only import (
    Int4WeightOnlyLinear,
    Int8WeightOnlyLinear,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is not visible"
)


@torch.inference_mode()
def test_weight_only_teaching_layers_execute_on_cuda():
    device = resolve_device("cuda")
    reference = torch.nn.Linear(7, 5, bias=True, device=device, dtype=torch.float32)
    inputs = torch.randn(3, 7, device=device)

    int8 = Int8WeightOnlyLinear(reference.weight, reference.bias)
    int4 = Int4WeightOnlyLinear(reference.weight, reference.bias)
    assert int8(inputs).device == device
    assert int4(inputs).device == device
    assert torch.allclose(int8(inputs), reference(inputs), atol=0.03, rtol=0.03)
    assert torch.allclose(int4(inputs), reference(inputs), atol=0.35, rtol=0.35)


def test_cuda_manifest_reports_visible_device():
    report = cuda_device_info("cuda")
    assert report["name"]
    assert report["total_memory_mib"] > 0
    assert report["compute_capability"]
