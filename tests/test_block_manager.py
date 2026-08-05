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
