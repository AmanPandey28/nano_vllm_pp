from types import SimpleNamespace

import torch

from nanovllm_pp.config import EngineConfig, ModelConfig


class TinyCacheModel(torch.nn.Module):
    """Deterministic causal-model double with a legacy tuple KV cache."""

    def __init__(self, transitions=None, vocab_size=32):
        super().__init__()
        self.transitions = transitions or {}
        self.vocab_size = vocab_size
        self.past_lengths = []

    @staticmethod
    def cache_length(cache):
        return 0 if cache is None else int(cache[0][0].shape[-2])

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        past_length = self.cache_length(past_key_values)
        self.past_lengths.append(past_length)
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocab_size), -100.0)
        for row in range(batch):
            for column in range(length):
                token = int(input_ids[row, column])
                predicted = self.transitions.get(token, (token + 1) % self.vocab_size)
                logits[row, column, predicted] = 100.0

        old_tokens = (
            torch.empty((batch, 1, 0, 1), dtype=torch.float32)
            if past_key_values is None
            else past_key_values[0][0]
        )
        new_tokens = input_ids.float().reshape(batch, 1, length, 1)
        states = torch.cat([old_tokens, new_tokens], dim=-2)
        cache = ((states, states.clone()),) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


def tiny_model_config(vocab_size=32):
    return ModelConfig(
        vocab_size=vocab_size,
        num_layers=1,
        num_kv_heads=1,
        num_attention_heads=1,
        head_dim=1,
        hidden_size=1,
        intermediate_size=1,
    )


def tiny_engine_config(**kwargs):
    defaults = {
        "block_size": 2,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 32,
        "max_model_len": 128,
    }
    defaults.update(kwargs)
    return EngineConfig(**defaults)
