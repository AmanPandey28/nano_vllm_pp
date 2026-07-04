"""Weight-only quantization for linear layers.

Implements per-channel INT8 and simulated INT4 quantization with naive
dequantization. The memory savings are real; speed improvements require
fused dequantize+matmul kernels (e.g., Triton, AWQ).
"""

import torch


def quantize_per_channel_int8(weight: torch.Tensor):
    """Quantize weight matrix to per-channel INT8.

    weight: [out_features, in_features]
    Returns (qweight [int8], scale [out_features]).

    Per-channel: each output channel gets its own scale factor, preserving
    more accuracy than a single per-tensor scale.
    """
    max_abs = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 127.0
    qweight = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return qweight, scale.squeeze(1)


def dequantize_int8(qweight: torch.Tensor, scale: torch.Tensor):
    """Dequantize per-channel INT8 back to float32."""
    return qweight.float() * scale[:, None]


def quantize_per_channel_int4(weight: torch.Tensor):
    """Quantize to 4-bit per-channel with packing (2 x int4 per byte)."""
    qweight, scale = quantize_per_channel_int8(weight)
    qweight_int4 = (qweight + 8).clamp(0, 15).to(torch.uint8)
    packed = (qweight_int4[::2] & 0x0F) | ((qweight_int4[1::2] & 0x0F) << 4)
    return packed, scale


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor):
    """Unpack 4-bit weights and dequantize."""
    low = (packed & 0x0F).float() - 8
    high = ((packed >> 4) & 0x0F).float() - 8
    qweight = torch.empty((packed.shape[0] * 2, packed.shape[1]), dtype=torch.float32)
    qweight[::2] = low
    qweight[1::2] = high
    return qweight * scale[:, None]


class Int8WeightOnlyLinear(torch.nn.Module):
    """Linear layer with quantized INT8 weights.

    Stores weights as int8 buffers with per-channel scale factors.
    Dequantizes on every forward pass — saves memory, not compute time.
    """

    def __init__(self, fp_weight: torch.Tensor, bias=None):
        super().__init__()
        qweight, scale = quantize_per_channel_int8(fp_weight.float())
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        weight = dequantize_int8(self.qweight, self.scale).to(x.dtype)
        out = x @ weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


def quantize_model_int8(model, exclude_keywords=("lm_head", "embed")):
    """Replace all nn.Linear layers with Int8WeightOnlyLinear.

    Skips embedding and lm_head layers by default, as these are sensitive
    to quantization and typically kept in full precision.

    Returns (num_quantized, num_skipped).
    """
    quantized_count = 0
    skipped_count = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(kw in name for kw in exclude_keywords):
            skipped_count += 1
            continue

        quantized = Int8WeightOnlyLinear(module.weight.data, module.bias)
        parent_name = ".".join(name.split(".")[:-1]) or ""
        child_name = name.split(".")[-1]
        parent = model if parent_name == "" else model.get_submodule(parent_name)
        setattr(parent, child_name, quantized)
        quantized_count += 1

    return quantized_count, skipped_count


def model_memory_report(model):
    """Report memory usage estimates for FP16, INT8, and INT4.

    Note: only counts named_parameters() (excludes buffers), so INT8/INT4
    values represent remaining FP16 parameters after quantization.
    """
    stats = {"FP16": 0, "INT8": 0, "INT4": 0}
    for name, param in model.named_parameters():
        elements = param.numel()
        stats["FP16"] += elements * 2
        stats["INT8"] += elements * 1
        stats["INT4"] += elements * 0.5

    return {
        "fp16_mb": round(stats["FP16"] / (1024**2), 1),
        "int8_mb": round(stats["INT8"] / (1024**2), 1),
        "int4_mb": round(stats["INT4"] / (1024**2), 1),
        "int8_vs_fp16": f"{stats['INT8'] / stats['FP16'] * 100:.0f}%",
        "int4_vs_fp16": f"{stats['INT4'] / stats['FP16'] * 100:.0f}%",
    }
