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
    max_abs = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 7.0
    qweight = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
    unsigned = (qweight + 8).to(torch.uint8)
    if unsigned.shape[1] % 2:
        # Unsigned nibble value 8 represents quantized zero.
        unsigned = torch.nn.functional.pad(unsigned, (0, 1), value=8)
    low = unsigned[:, 0::2] & 0x0F
    high = (unsigned[:, 1::2] & 0x0F) << 4
    packed = low | high
    return packed, scale.squeeze(1)


def dequantize_int4(
    packed: torch.Tensor, scale: torch.Tensor, in_features: int | None = None
):
    """Unpack 4-bit weights and dequantize."""
    low = (packed & 0x0F).float() - 8
    high = ((packed >> 4) & 0x0F).float() - 8
    qweight = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=torch.float32,
        device=packed.device,
    )
    qweight[:, 0::2] = low
    qweight[:, 1::2] = high
    if in_features is not None:
        qweight = qweight[:, :in_features]
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
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x):
        weight = dequantize_int8(self.qweight, self.scale).to(x.dtype)
        out = x @ weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Int4WeightOnlyLinear(torch.nn.Module):
    """Correct packed INT4 storage with naive dequantization on each forward."""

    def __init__(self, fp_weight: torch.Tensor, bias=None):
        super().__init__()
        self.out_features, self.in_features = fp_weight.shape
        packed, scale = quantize_per_channel_int4(fp_weight.float())
        self.register_buffer("packed_weight", packed)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x):
        weight = dequantize_int4(
            self.packed_weight, self.scale, self.in_features
        ).to(x.dtype)
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


def quantize_model_int4(model, exclude_keywords=("lm_head", "embed")):
    """Replace eligible ``nn.Linear`` modules with packed INT4 prototypes."""
    quantized_count = 0
    skipped_count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(keyword in name for keyword in exclude_keywords):
            skipped_count += 1
            continue
        quantized = Int4WeightOnlyLinear(module.weight.data, module.bias)
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)
        setattr(parent, child_name, quantized)
        quantized_count += 1
    return quantized_count, skipped_count


def model_memory_report(model):
    """Report actual parameter and buffer storage, deduplicating tied tensors."""
    seen = set()
    parameter_bytes = 0
    buffer_bytes = 0
    for tensor, kind in [
        *((tensor, "parameter") for tensor in model.parameters()),
        *((tensor, "buffer") for tensor in model.buffers()),
    ]:
        if tensor.device.type == "meta":
            continue
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        size = tensor.numel() * tensor.element_size()
        if kind == "parameter":
            parameter_bytes += size
        else:
            buffer_bytes += size
    total = parameter_bytes + buffer_bytes
    return {
        "actual_bytes": total,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "actual_mb": round(total / (1024**2), 1),
        "parameter_mb": round(parameter_bytes / (1024**2), 1),
        "buffer_mb": round(buffer_bytes / (1024**2), 1),
    }
