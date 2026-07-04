from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    block_size: int = 256
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = False
    max_prefill_chunk_size: int = 512
    decode_priority: bool = True
    reserve_full_prompt_blocks: bool = True

    enable_spec_decode: bool = False
    max_spec_tokens: int = 4
    spec_proposer_type: str = "ngram"
    reserve_lookahead_slots: bool = True

    def __post_init__(self):
        assert self.block_size > 0
        assert self.max_num_seqs > 0
        assert self.max_num_batched_tokens > 0
        assert 0.0 < self.gpu_memory_utilization <= 1.0


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
