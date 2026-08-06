import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in CPU-only installs
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _store_kv_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        slot_ptr,
        stride_token,
        stride_head,
        cache_stride_slot,
        cache_stride_head,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_index = tl.program_id(0)
        head_index = tl.program_id(1)
        dimensions = tl.arange(0, BLOCK_D)
        mask = dimensions < HEAD_DIM
        slot = tl.load(slot_ptr + token_index)

        source = token_index * stride_token + head_index * stride_head + dimensions
        target = slot * cache_stride_slot + head_index * cache_stride_head + dimensions
        key = tl.load(key_ptr + source, mask=mask)
        value = tl.load(value_ptr + source, mask=mask)
        tl.store(key_cache_ptr + target, key, mask=mask)
        tl.store(value_cache_ptr + target, value, mask=mask)

    @triton.jit
    def _paged_decode_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_table_ptr,
        context_lens_ptr,
        output_ptr,
        stride_query_batch,
        stride_query_head,
        cache_stride_slot,
        cache_stride_head,
        stride_table_batch,
        stride_output_batch,
        stride_output_head,
        SCALE: tl.constexpr,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MAX_CONTEXT: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        query_head = tl.program_id(1)
        kv_head = query_head // (NUM_QUERY_HEADS // NUM_KV_HEADS)
        dimensions = tl.arange(0, BLOCK_D)
        dimension_mask = dimensions < HEAD_DIM
        query = tl.load(
            query_ptr
            + batch_index * stride_query_batch
            + query_head * stride_query_head
            + dimensions,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

        context_length = tl.load(context_lens_ptr + batch_index)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for token_start in range(0, MAX_CONTEXT, BLOCK_N):
            token_offsets = token_start + tl.arange(0, BLOCK_N)
            token_mask = token_offsets < context_length
            logical_blocks = token_offsets // BLOCK_SIZE
            physical_blocks = tl.load(
                block_table_ptr + batch_index * stride_table_batch + logical_blocks,
                mask=token_mask,
                other=0,
            )
            slots = physical_blocks * BLOCK_SIZE + token_offsets % BLOCK_SIZE
            cache_offsets = (
                slots[:, None] * cache_stride_slot
                + kv_head * cache_stride_head
                + dimensions[None, :]
            )
            keys = tl.load(
                key_cache_ptr + cache_offsets,
                mask=token_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * SCALE
            scores = tl.where(token_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value_cache_ptr + cache_offsets,
                mask=token_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(
                probabilities[:, None] * values, axis=0
            )
            running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
            running_max = next_max

        result = accumulator / running_sum
        tl.store(
            output_ptr
            + batch_index * stride_output_batch
            + query_head * stride_output_head
            + dimensions,
            result,
            mask=dimension_mask,
        )


def store_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter new ``[token, kv_head, dim]`` states into physical pages."""
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key and value must have equal [tokens, heads, dim] shapes")
    if slot_mapping.numel() != key.shape[0]:
        raise ValueError("slot_mapping must contain one slot per token")

    flat_key_cache = key_cache.view(-1, key_cache.shape[-2], key_cache.shape[-1])
    flat_value_cache = value_cache.view_as(flat_key_cache)
    slots = slot_mapping.to(device=key.device, dtype=torch.long)
    if key.is_cuda and triton is not None:
        head_dim = key.shape[-1]
        block_d = triton.next_power_of_2(head_dim)
        _store_kv_kernel[(key.shape[0], key.shape[1])](
            key,
            value,
            flat_key_cache,
            flat_value_cache,
            slots,
            key.stride(0),
            key.stride(1),
            flat_key_cache.stride(0),
            flat_key_cache.stride(1),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
    else:
        flat_key_cache.index_copy_(0, slots, key)
        flat_value_cache.index_copy_(0, slots, value)


def gather_paged_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_length: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one logical context from the global physical page pool."""
    positions = torch.arange(context_length, device=key_cache.device)
    page_indices = torch.div(positions, block_size, rounding_mode="floor")
    physical_pages = block_table.index_select(0, page_indices).long()
    slots = physical_pages * block_size + positions.remainder(block_size)
    flat_keys = key_cache.view(-1, key_cache.shape[-2], key_cache.shape[-1])
    flat_values = value_cache.view_as(flat_keys)
    return flat_keys.index_select(0, slots), flat_values.index_select(0, slots)


def paged_prefill_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """Exact SDPA prefill over physical pages, including cached prefixes.

    This readable reference path materializes each request's logical K/V view.
    Decode—the latency-critical path—reads pages directly in Triton below.
    """
    outputs = []
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[-2]
    if num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    repeats = num_query_heads // num_kv_heads

    for batch_index in range(metadata.num_sequences):
        query_start = metadata.query_starts[batch_index]
        query_end = metadata.query_starts[batch_index + 1]
        context_length = metadata.context_lengths[batch_index]
        query_slice = query[query_start:query_end]
        keys, values = gather_paged_kv(
            key_cache,
            value_cache,
            metadata.block_tables[batch_index],
            context_length,
            metadata.block_size,
        )
        keys = keys.repeat_interleave(repeats, dim=1)
        values = values.repeat_interleave(repeats, dim=1)

        absolute_queries = metadata.positions[query_start:query_end]
        key_positions = torch.arange(context_length, device=query.device)
        causal_mask = key_positions[None, :] <= absolute_queries[:, None]
        attended = F.scaled_dot_product_attention(
            query_slice.transpose(0, 1).unsqueeze(0),
            keys.transpose(0, 1).unsqueeze(0),
            values.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_mask.unsqueeze(0).unsqueeze(0),
            dropout_p=0.0,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def paged_decode_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """PyTorch correctness oracle for the direct paged decode kernel."""
    outputs = []
    num_query_heads = query.shape[1]
    repeats = num_query_heads // key_cache.shape[-2]
    scale = 1.0 / math.sqrt(query.shape[-1])
    for batch_index, context_length_tensor in enumerate(context_lens):
        context_length = int(context_length_tensor.item())
        keys, values = gather_paged_kv(
            key_cache,
            value_cache,
            block_tables[batch_index],
            context_length,
            block_size,
        )
        keys = keys.repeat_interleave(repeats, dim=1)
        values = values.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,thd->ht", query[batch_index], keys) * scale
        probabilities = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        outputs.append(torch.einsum("ht,thd->hd", probabilities, values))
    return torch.stack(outputs)


def paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int | None = None,
) -> torch.Tensor:
    """Decode attention that follows block tables without gathering K/V."""
    if not query.is_cuda or triton is None:
        return paged_decode_attention_reference(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            block_size,
        )
    if query.shape[1] % key_cache.shape[-2]:
        raise ValueError("query heads must be divisible by KV heads")
    if query.shape[-1] > 256:
        raise ValueError("the teaching Triton kernel supports head_dim <= 256")

    output = torch.empty_like(query)
    if max_context_len is None:
        max_context_len = int(context_lens.max().item())
    maximum_context = max(16, triton.next_power_of_2(max_context_len))
    block_d = triton.next_power_of_2(query.shape[-1])
    flat_keys = key_cache.view(-1, key_cache.shape[-2], key_cache.shape[-1])
    flat_values = value_cache.view_as(flat_keys)
    _paged_decode_kernel[(query.shape[0], query.shape[1])](
        query,
        flat_keys,
        flat_values,
        block_tables,
        context_lens,
        output,
        query.stride(0),
        query.stride(1),
        flat_keys.stride(0),
        flat_keys.stride(1),
        block_tables.stride(0),
        output.stride(0),
        output.stride(1),
        SCALE=1.0 / math.sqrt(query.shape[-1]),
        NUM_QUERY_HEADS=query.shape[1],
        NUM_KV_HEADS=key_cache.shape[-2],
        HEAD_DIM=query.shape[-1],
        BLOCK_SIZE=block_size,
        BLOCK_D=block_d,
        BLOCK_N=32,
        MAX_CONTEXT=maximum_context,
        num_warps=4,
    )
    return output
