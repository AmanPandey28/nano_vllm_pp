from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig

from .kernels import paged_decode_attention, paged_prefill_attention, store_kv


def _rope_parameters(config):
    parameters = getattr(config, "rope_parameters", None)
    if parameters is None:
        parameters = getattr(config, "rope_scaling", None)
    return parameters or {}


def _rope_theta(config):
    return getattr(config, "rope_theta", None) or _rope_parameters(config).get(
        "rope_theta", 10000.0
    )


class Qwen3RMSNorm(nn.Module):
    """RMSNorm with float32 accumulation and a learned per-channel scale."""

    def __init__(self, hidden_size, eps=1e-6, *, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        values = values * torch.rsqrt(variance + self.eps)
        return self.weight * values.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    """Qwen's half-split rotary position embedding."""

    def __init__(self, head_dim, theta, *, device=None):
        super().__init__()
        frequencies = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def forward(self, positions, dtype):
        frequencies = torch.outer(positions.float(), self.inv_freq.float())
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _rotate_half(values):
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(query, key, cosine, sine):
    cosine = cosine[:, None, :]
    sine = sine[:, None, :]
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


class Qwen3Attention(nn.Module):
    def __init__(self, config, *, device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        if self.num_heads % self.num_kv_heads:
            raise ValueError("Qwen query heads must be divisible by KV heads")
        projection = {"bias": False, "device": device, "dtype": dtype}
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, **projection
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, **projection
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, **projection
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, **projection
        )
        self.q_norm = Qwen3RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.k_norm = Qwen3RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.rotary = Qwen3RotaryEmbedding(
            self.head_dim, _rope_theta(config), device=device
        )

    def forward(self, hidden_states, metadata, key_cache, value_cache):
        query = self.q_proj(hidden_states).reshape(-1, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(-1, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(-1, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        cosine, sine = self.rotary(metadata.positions, query.dtype)
        query, key = _apply_rotary(query, key, cosine, sine)

        # This is the physical cache write. Every layer uses the same slot map
        # but owns a different slice of the global [layer, page, ...] pool.
        store_kv(
            key.contiguous(),
            value.contiguous(),
            key_cache,
            value_cache,
            metadata.slot_mapping,
        )
        if metadata.is_decode:
            attended = paged_decode_attention(
                query,
                key_cache,
                value_cache,
                metadata.block_tables,
                metadata.context_lens,
                metadata.block_size,
                metadata.max_context_len,
            )
        else:
            attended = paged_prefill_attention(query, key_cache, value_cache, metadata)
        return self.o_proj(attended.reshape(-1, self.num_heads * self.head_dim))


class Qwen3MLP(nn.Module):
    def __init__(self, config, *, device=None, dtype=None):
        super().__init__()
        projection = {"bias": False, "device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, **projection
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, **projection
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, **projection
        )

    def forward(self, hidden_states):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, *, device=None, dtype=None):
        super().__init__()
        self.self_attn = Qwen3Attention(config, device=device, dtype=dtype)
        self.mlp = Qwen3MLP(config, device=device, dtype=dtype)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states, metadata, key_cache, value_cache):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states, metadata, key_cache, value_cache
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class Qwen3Model(nn.Module):
    def __init__(self, config, *, device=None, dtype=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, device=device, dtype=dtype)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_ids, metadata, kv_cache):
        hidden_states = self.embed_tokens(input_ids)
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                metadata,
                kv_cache[0, layer_index],
                kv_cache[1, layer_index],
            )
        return self.norm(hidden_states)


class PagedQwen3ForCausalLM(nn.Module):
    """Minimal Qwen3 inference model whose attention consumes physical pages."""

    def __init__(self, config, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, metadata, kv_cache, logit_indices=None):
        hidden_states = self.model(input_ids, metadata, kv_cache)
        if logit_indices is not None:
            hidden_states = hidden_states.index_select(0, logit_indices)
        return self.lm_head(hidden_states)


def _weight_files(model_path: Path) -> list[Path]:
    safetensors = sorted(model_path.glob("*.safetensors"))
    if safetensors:
        return safetensors
    binaries = sorted(model_path.glob("pytorch_model*.bin"))
    if binaries:
        return binaries
    raise FileNotFoundError(f"no safetensors or PyTorch checkpoint in {model_path}")


def load_paged_qwen3(model_path, device, dtype):
    """Construct the custom model and stream Hugging Face weights into it."""
    from .kernels import triton

    if triton is None:
        raise RuntimeError(
            "model_backend='paged' requires Triton; install the 'paged' extra"
        )
    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = set(getattr(config, "architectures", []) or [])
    if "Qwen3ForCausalLM" not in architectures:
        raise NotImplementedError(
            "the physical paged backend currently supports Qwen3ForCausalLM; "
            f"checkpoint declares {sorted(architectures) or 'no architecture'}"
        )
    if getattr(config, "attention_bias", False):
        raise NotImplementedError(
            "Qwen checkpoints with attention bias are unsupported"
        )
    if getattr(config, "hidden_act", "silu") != "silu":
        raise NotImplementedError("the paged Qwen MLP currently implements SiLU")
    layer_types = getattr(config, "layer_types", None) or []
    if getattr(config, "use_sliding_window", False) or any(
        layer_type != "full_attention" for layer_type in layer_types
    ):
        raise NotImplementedError(
            "sliding-window attention is not implemented by the paged backend"
        )
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    if head_dim % 2 or head_dim > 256:
        raise NotImplementedError(
            "the paged Qwen backend requires an even head_dim no larger than 256"
        )
    rope_type = _rope_parameters(config).get("rope_type", "default")
    if rope_type != "default":
        raise NotImplementedError(
            "Qwen checkpoints with RoPE scaling are not supported by this backend"
        )

    model = PagedQwen3ForCausalLM(config, device=torch.device(device), dtype=dtype)
    targets = dict(model.named_parameters(remove_duplicate=False))
    loaded = set()
    files = _weight_files(model_path)
    with torch.inference_mode():
        for checkpoint in files:
            if checkpoint.suffix == ".safetensors":
                from safetensors import safe_open

                with safe_open(checkpoint, framework="pt", device="cpu") as handle:
                    # safe_open exposes keys() but is not a Mapping/iterator.
                    for name in handle.keys():  # noqa: SIM118
                        target = targets.get(name)
                        if target is None:
                            continue
                        target.copy_(
                            handle.get_tensor(name).to(
                                device=target.device, dtype=target.dtype
                            )
                        )
                        loaded.add(name)
            else:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                for name, value in state.items():
                    target = targets.get(name)
                    if target is None:
                        continue
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
                    loaded.add(name)

    missing = set(targets) - loaded
    if (
        config.tie_word_embeddings
        and {
            "model.embed_tokens.weight",
            "lm_head.weight",
        }
        & loaded
    ):
        # Either checkpoint spelling initializes the shared Parameter object.
        missing -= {"model.embed_tokens.weight", "lm_head.weight"}
    if missing:
        preview = ", ".join(sorted(missing)[:8])
        raise RuntimeError(f"checkpoint is missing paged-model weights: {preview}")
    model.eval()
    return model, config
