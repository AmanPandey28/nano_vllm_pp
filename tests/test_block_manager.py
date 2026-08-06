from nanovllm_pp.engine.block_manager import BlockManager, compute_block_hash
from nanovllm_pp.engine.sequence import Sequence
from nanovllm_pp.sampling_params import SamplingParams


def make_sequence(tokens):
    return Sequence(tokens, SamplingParams(max_tokens=8))


def test_chained_hash_depends_on_parent_and_tokens():
    assert compute_block_hash(None, [1, 2]) == compute_block_hash(None, [1, 2])
    assert compute_block_hash(None, [1, 2]) != compute_block_hash(None, [2, 1])
    assert compute_block_hash("parent-a", [1, 2]) != compute_block_hash(
        "parent-b", [1, 2]
    )


def test_allocate_append_and_deallocate_accounting():
    manager = BlockManager(num_blocks=5, block_size=2)
    seq = make_sequence([1, 2, 3])

    manager.allocate(seq)
    assert len(seq.block_table) == 2
    assert manager.num_free_blocks == 3

    assert manager.can_append(seq, num_new_tokens=4)
    assert manager.may_append(seq, num_new_tokens=4)
    assert len(seq.block_table) == 4
    assert manager.num_free_blocks == 1

    manager.deallocate(seq)
    assert seq.block_table == []
    assert manager.num_free_blocks == 5


def test_reused_free_block_evicts_stale_hash_mapping():
    manager = BlockManager(num_blocks=1, block_size=2, enable_prefix_caching=True)
    first = make_sequence([1, 2])
    manager.allocate(first)
    old_hash = manager.blocks[first.block_table[0]].hash
    manager.deallocate(first)

    second = make_sequence([9])
    manager.allocate(second)
    assert old_hash not in manager.hash_to_block_id


def test_block_table_translates_positions_to_physical_slots():
    manager = BlockManager(num_blocks=6, block_size=2)
    seq = make_sequence([10, 11, 12])
    manager.allocate(seq, num_new_tokens=2)

    expected = [
        seq.block_table[0] * 2,
        seq.block_table[0] * 2 + 1,
        seq.block_table[1] * 2,
        seq.block_table[1] * 2 + 1,
        seq.block_table[2] * 2,
    ]
    assert manager.slot_mapping(seq, range(5)) == expected


def test_generated_tokens_fill_reserved_block_metadata():
    manager = BlockManager(num_blocks=4, block_size=2)
    seq = make_sequence([10, 11])
    manager.allocate(seq, num_new_tokens=2)

    seq.append_token(12)
    manager.record_token(seq, 12)
    seq.append_token(13)
    manager.record_token(seq, 13)

    generated_block = manager.blocks[seq.block_table[1]]
    assert generated_block.token_ids == [12, 13]
    assert generated_block.hash is not None


def test_prompt_boundary_reserves_first_generated_token_slot():
    manager = BlockManager(num_blocks=2, block_size=2)
    seq = make_sequence([1, 2])

    assert manager.can_allocate(seq, num_new_tokens=1)
    manager.allocate(seq, num_new_tokens=1)
    assert len(seq.block_table) == 2

    seq.append_token(3)
    manager.record_token(seq, 3)
    assert manager.slot_mapping(seq, [2]) == [seq.block_table[1] * 2]


def test_prefix_hash_is_published_only_after_physical_page_is_complete():
    manager = BlockManager(num_blocks=6, block_size=2, enable_prefix_caching=True)
    first = make_sequence([1, 2, 3])
    manager.allocate(first)
    first_page = manager.blocks[first.block_table[0]]

    assert first_page.hash is not None
    assert first_page.hash not in manager.hash_to_block_id
    manager.mark_computed(first, 0, 1)
    assert first_page.hash not in manager.hash_to_block_id
    manager.mark_computed(first, 1, 1)
    assert manager.hash_to_block_id[first_page.hash] == first_page.block_id


def test_prefix_allocation_reuses_physically_computed_pages():
    manager = BlockManager(num_blocks=8, block_size=2, enable_prefix_caching=True)
    first = make_sequence([1, 2, 3, 4, 5])
    manager.allocate(first)
    manager.mark_computed(first, 0, first.prompt_len)
    shared_pages = first.block_table[:2]
    manager.deallocate(first)

    second = make_sequence([1, 2, 3, 4, 9])
    assert manager.cached_prefix_tokens(second) == 4
    manager.allocate(second)

    assert second.block_table[:2] == shared_pages
    assert second.num_cached_tokens == 4
    assert manager.blocks[shared_pages[0]].computed_tokens == 2


def test_exact_block_boundary_keeps_one_query_block_for_first_token_logits():
    manager = BlockManager(num_blocks=8, block_size=2, enable_prefix_caching=True)
    first = make_sequence([1, 2, 3, 4])
    manager.allocate(first)
    manager.mark_computed(first, 0, first.prompt_len)
    manager.deallocate(first)

    second = make_sequence([1, 2, 3, 4])
    assert manager.cached_prefix_tokens(second) == 2
