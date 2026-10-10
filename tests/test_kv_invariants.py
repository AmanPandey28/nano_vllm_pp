"""Boundary and randomized KV ownership audits, independent of model weights."""

import random
from collections import Counter

import pytest

from nanovllm_pp.engine.block_manager import BlockManager, compute_block_hash
from nanovllm_pp.engine.sequence import Sequence
from nanovllm_pp.sampling_params import SamplingParams


def sequence(tokens):
    return Sequence(tokens, SamplingParams(temperature=0, max_tokens=8))


@pytest.mark.parametrize("block_size", [1, 2, 4])
@pytest.mark.parametrize("length", [0, 1, 2, 4, 5, 11])
def test_boundary_ownership_and_full_page_publication(block_size, length):
    manager = BlockManager(32, block_size, enable_prefix_caching=True)
    seq = sequence(list(range(length)))
    manager.allocate(seq, num_new_tokens=1)
    manager.mark_computed(seq, 0, length)
    manager.assert_invariants([seq])
    assert len(manager.hash_to_block_id) == length // block_size
    manager.deallocate(seq)
    manager.deallocate(seq)  # Empty-table release is idempotent.
    manager.assert_invariants([])


@pytest.mark.parametrize("seed", [7, 17, 29, 43, 61])
def test_randomized_share_append_free_eviction_invariants(seed):
    rng = random.Random(seed)
    manager = BlockManager(20, 2, enable_prefix_caching=True)
    live = []
    for _ in range(300):
        if live and rng.random() < 0.45:
            seq = live.pop(rng.randrange(len(live)))
            manager.deallocate(seq)
        else:
            tokens = [1, 2, 3, 4, rng.randrange(5, 15)]
            seq = sequence(tokens)
            if manager.can_allocate(seq, num_new_tokens=2):
                manager.allocate(seq, num_new_tokens=2)
                live.append(seq)
                manager.mark_computed(
                    seq, seq.num_cached_tokens, seq.prompt_len - seq.num_cached_tokens
                )
                for _ in range(rng.randrange(3)):
                    if manager.may_append(seq):
                        seq.append_token(rng.randrange(20, 30))
                        manager.record_token(seq, seq.last_token)
                        manager.mark_computed(seq, seq.num_tokens - 1, 1)
        counts = Counter(page for seq in live for page in seq.block_table)
        assert all(
            block.ref_count == counts[block.block_id] for block in manager.blocks
        )
        manager.assert_invariants(live)
    for seq in live:
        manager.deallocate(seq)
    manager.assert_invariants([])


def test_failed_multi_page_append_is_atomic():
    manager = BlockManager(2, 2)
    seq = sequence([1])
    manager.allocate(seq)
    before = list(seq.block_table), set(manager.free_block_ids)
    assert not manager.may_append(seq, num_new_tokens=5)
    assert (seq.block_table, manager.free_block_ids) == before
    manager.assert_invariants([seq])


def test_live_shared_page_cannot_be_evicted_or_mutated():
    manager = BlockManager(4, 2, enable_prefix_caching=True)
    first = sequence([1, 2, 3])
    manager.allocate(first)
    manager.mark_computed(first, 0, 3)
    second = sequence([1, 2, 9])
    manager.allocate(second)
    shared = first.block_table[0]
    assert second.block_table[0] == shared
    assert manager.blocks[shared].ref_count == 2
    third = sequence([8])
    manager.allocate(third)
    assert shared not in third.block_table
    with pytest.raises(RuntimeError, match="overwrite"):
        manager.record_tokens(second, 0, [99])
    manager.deallocate(first)
    assert manager.blocks[shared].ref_count == 1
    assert shared not in manager.free_block_ids
    manager.assert_invariants([second, third])


def test_identical_block_tokens_with_different_parent_do_not_share():
    manager = BlockManager(12, 2, enable_prefix_caching=True)
    first = sequence([1, 2, 7, 8, 9])
    manager.allocate(first)
    manager.mark_computed(first, 0, 5)
    second = sequence([3, 4, 7, 8, 9])
    assert manager.cached_prefix_tokens(second) == 0
    assert compute_block_hash(compute_block_hash(None, [1, 2]), [7, 8]) != (
        compute_block_hash(compute_block_hash(None, [3, 4]), [7, 8])
    )


def test_lookup_rejects_hash_entry_with_wrong_local_tokens():
    manager = BlockManager(4, 2, enable_prefix_caching=True)
    seq = sequence([1, 2, 3])
    manager.allocate(seq)
    manager.mark_computed(seq, 0, 3)
    manager.blocks[seq.block_table[0]].token_ids = [8, 9]
    assert manager.cached_prefix_tokens(sequence([1, 2, 4])) == 0


def test_recycled_page_invalidates_old_prefix_index():
    manager = BlockManager(2, 2, enable_prefix_caching=True)
    first = sequence([1, 2, 3])
    manager.allocate(first)
    manager.mark_computed(first, 0, 3)
    manager.deallocate(first)
    replacement = sequence([7, 8, 9])
    manager.allocate(replacement)
    manager.mark_computed(replacement, 0, 3)
    assert manager.cached_prefix_tokens(sequence([1, 2, 4])) == 0
    assert manager.stats()["evictions"] == 1
    manager.assert_invariants([replacement])
