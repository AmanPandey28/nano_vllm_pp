import torch

from nanovllm_pp.quantization.int8_weight_only import (
    Int4WeightOnlyLinear,
    dequantize_int4,
    model_memory_report,
    quantize_per_channel_int4,
    quantize_per_channel_int8,
)


def test_int4_round_trip_handles_odd_input_width_and_error_bound():
    weight = torch.tensor(
        [[-1.0, -0.7, 0.0, 0.3, 1.0], [0.2, -0.4, 0.6, 0.8, -1.0]]
    )
    packed, scale = quantize_per_channel_int4(weight)
    restored = dequantize_int4(packed, scale, in_features=weight.shape[1])

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 3)
    assert restored.shape == weight.shape
    max_error = torch.max(torch.abs(restored - weight), dim=1).values
    assert torch.all(max_error <= scale / 2 + 1e-6)


def test_int4_linear_preserves_shape_and_registers_storage_as_buffers():
    layer = Int4WeightOnlyLinear(torch.randn(3, 5), bias=torch.randn(3))
    output = layer(torch.randn(2, 5))

    assert output.shape == (2, 3)
    assert list(layer.parameters()) == []
    assert {name for name, _ in layer.named_buffers()} == {
        "packed_weight",
        "scale",
        "bias",
    }


def test_int8_memory_report_counts_quantized_buffers():
    linear = torch.nn.Linear(8, 4, bias=False)
    qweight, scale = quantize_per_channel_int8(linear.weight.detach())
    assert qweight.numel() == 32
    assert scale.numel() == 4

    quantized = __import__(
        "nanovllm_pp.quantization.int8_weight_only", fromlist=["Int8WeightOnlyLinear"]
    ).Int8WeightOnlyLinear(linear.weight.detach())
    report = model_memory_report(quantized)
    assert report["parameter_bytes"] == 0
    assert report["buffer_bytes"] == qweight.numel() + scale.numel() * 4
    assert report["actual_bytes"] < linear.weight.numel() * 4
