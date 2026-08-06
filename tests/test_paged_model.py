import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from nanovllm_pp.paged.context import PagedBatchMetadata
from nanovllm_pp.paged.model import PagedQwen3ForCausalLM


def metadata(positions, slots, table, context_length, *, is_decode):
    return PagedBatchMetadata(
        positions=torch.tensor(positions, dtype=torch.long),
        slot_mapping=torch.tensor(slots, dtype=torch.long),
        query_start_loc=torch.tensor([0, len(positions)], dtype=torch.int32),
        context_lens=torch.tensor([context_length], dtype=torch.int32),
        block_tables=torch.tensor([table], dtype=torch.int32),
        is_decode=is_decode,
        block_size=2,
        query_starts=(0, len(positions)),
        context_lengths=(context_length,),
        max_context_len=context_length,
    )


@torch.inference_mode()
def test_tiny_paged_qwen_matches_hugging_face_prefill_and_decode():
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=True,
    )
    reference = Qwen3ForCausalLM(config).eval()
    paged = PagedQwen3ForCausalLM(config, dtype=torch.float32).eval()
    paged.load_state_dict(reference.state_dict(), strict=True)
    cache = torch.empty(2, 2, 4, 2, 2, 4)

    prompt = torch.tensor([[3, 7, 11, 5]])
    reference_prefill = reference(prompt, use_cache=True)
    paged_prefill = paged(
        prompt[0],
        metadata([0, 1, 2, 3], [4, 5, 0, 1], [2, 0], 4, is_decode=False),
        cache,
    )
    assert torch.allclose(paged_prefill, reference_prefill.logits[0], atol=2e-5)

    next_token = torch.tensor([[13]])
    reference_decode = reference(
        next_token,
        past_key_values=reference_prefill.past_key_values,
        use_cache=True,
    )
    paged_decode = paged(
        next_token[0],
        metadata([4], [6], [2, 0, 3], 5, is_decode=True),
        cache,
    )
    assert torch.allclose(paged_decode, reference_decode.logits[0], atol=2e-5)
