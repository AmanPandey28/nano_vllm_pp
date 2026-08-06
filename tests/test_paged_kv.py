import math

import pytest
import torch

from nanovllm_pp.paged.context import PagedBatchMetadata
from nanovllm_pp.paged.kernels import (
    gather_paged_kv,
    paged_decode_attention,
    paged_decode_attention_reference,
    paged_prefill_attention,
    store_kv,
)


def test_store_and_gather_follow_noncontiguous_block_table():
    key_cache = torch.zeros(4, 2, 1, 3)
    value_cache = torch.zeros_like(key_cache)
    keys = torch.arange(15, dtype=torch.float32).reshape(5, 1, 3)
    values = keys + 100
    # Logical pages 0, 1, 2 live in physical pages 3, 0, 2.
    block_table = torch.tensor([3, 0, 2], dtype=torch.int32)
    slots = torch.tensor([6, 7, 0, 1, 4])

    store_kv(keys, values, key_cache, value_cache, slots)
    gathered_keys, gathered_values = gather_paged_kv(
        key_cache, value_cache, block_table, context_length=5, block_size=2
    )

    assert torch.equal(gathered_keys, keys)
    assert torch.equal(gathered_values, values)


def test_decode_reference_handles_grouped_query_attention():
    torch.manual_seed(7)
    query = torch.randn(2, 4, 8)
    key_cache = torch.randn(5, 2, 2, 8)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.tensor([[3, 0, 4], [1, 2, 0]], dtype=torch.int32)
    context_lens = torch.tensor([5, 3], dtype=torch.int32)

    output = paged_decode_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, block_size=2
    )

    for batch_index, context_length in enumerate([5, 3]):
        keys, values = gather_paged_kv(
            key_cache,
            value_cache,
            block_tables[batch_index],
            context_length,
            2,
        )
        keys = keys.repeat_interleave(2, dim=1)
        values = values.repeat_interleave(2, dim=1)
        scores = torch.einsum("hd,thd->ht", query[batch_index], keys) / math.sqrt(8)
        expected = torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), values)
        assert torch.allclose(output[batch_index], expected)


def test_prefill_attention_reads_cached_prefix_and_applies_absolute_causality():
    torch.manual_seed(9)
    query = torch.randn(2, 4, 8)
    key_cache = torch.randn(3, 2, 2, 8)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32)
    metadata = PagedBatchMetadata(
        positions=torch.tensor([2, 3]),
        slot_mapping=torch.tensor([0, 1]),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        context_lens=torch.tensor([4], dtype=torch.int32),
        block_tables=block_table,
        is_decode=False,
        block_size=2,
        query_starts=(0, 2),
        context_lengths=(4,),
        max_context_len=4,
    )

    output = paged_prefill_attention(query, key_cache, value_cache, metadata)
    keys, values = gather_paged_kv(
        key_cache, value_cache, block_table[0], context_length=4, block_size=2
    )
    keys = keys.repeat_interleave(2, dim=1)
    values = values.repeat_interleave(2, dim=1)
    for query_index, absolute_position in enumerate([2, 3]):
        scores = torch.einsum(
            "hd,thd->ht", query[query_index], keys[: absolute_position + 1]
        ) / math.sqrt(8)
        expected = torch.einsum(
            "ht,thd->hd",
            torch.softmax(scores, dim=-1),
            values[: absolute_position + 1],
        )
        assert torch.allclose(output[query_index], expected, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not visible")
def test_triton_decode_matches_torch_reference_on_cuda():
    torch.manual_seed(11)
    device = torch.device("cuda")
    query = torch.randn(2, 4, 128, device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(7, 4, 2, 128, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.tensor(
        [[5, 1, 6, 0], [3, 2, 4, 1]], dtype=torch.int32, device=device
    )
    context_lens = torch.tensor([13, 9], dtype=torch.int32, device=device)

    expected = paged_decode_attention_reference(
        query, key_cache, value_cache, block_tables, context_lens, block_size=4
    )
    actual = paged_decode_attention(
        query, key_cache, value_cache, block_tables, context_lens, block_size=4
    )

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
