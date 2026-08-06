from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    device: str = "auto"
    dtype: str = "auto"
    model_backend: str = "hf"
    block_size: int = 256
    kv_cache_num_blocks: int | None = None
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    # Physical prefix reuse is available with model_backend="paged". The HF
    # compatibility backend rejects it because its per-request DynamicCache
    # cannot safely alias pages owned by BlockManager.
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    max_prefill_chunk_size: int = 512
    decode_priority: bool = True

    enable_spec_decode: bool = False
    max_spec_tokens: int = 4
    spec_proposer_type: str = "ngram"

    def __post_init__(self):
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be one of: auto, float32, float16, bfloat16")
        if self.model_backend not in {"hf", "paged"}:
            raise ValueError("model_backend must be one of: hf, paged")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.kv_cache_num_blocks is not None and self.kv_cache_num_blocks <= 0:
            raise ValueError("kv_cache_num_blocks must be > 0 or None")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be > 0")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be > 0")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_prefill_chunk_size <= 0:
            raise ValueError("max_prefill_chunk_size must be > 0")
        if self.max_spec_tokens <= 0:
            raise ValueError("max_spec_tokens must be > 0")
        if self.spec_proposer_type != "ngram":
            raise ValueError("only the ngram speculative proposer is supported")
        if self.enable_spec_decode and self.max_num_batched_tokens < 2:
            raise ValueError("speculative decoding needs a token budget of at least 2")


@dataclass
class ModelConfig:
    model_path: str = ""
    num_layers: int = 0
    num_kv_heads: int = 0
    num_attention_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    dtype: str = "auto"
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    extra: dict = field(default_factory=dict)
