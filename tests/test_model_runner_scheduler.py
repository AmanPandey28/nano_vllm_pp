from conftest import TinyCacheModel, tiny_engine_config, tiny_model_config

from nanovllm_pp.engine.block_manager import BlockManager
from nanovllm_pp.engine.model_runner import ModelRunner
from nanovllm_pp.engine.scheduler import Scheduler
from nanovllm_pp.engine.sequence import Sequence, SequenceStatus
from nanovllm_pp.sampling_params import SamplingParams


def test_chunked_prefill_carries_cache_between_chunks():
    model = TinyCacheModel()
    runner = ModelRunner(
        model, tiny_model_config(), tiny_engine_config(), device="cpu"
    )
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(temperature=0, max_tokens=3))

    assert runner.run_prefill(seq, 2) is None
    seq.num_computed_tokens += 2
    assert runner.run_prefill(seq, 2) is None
    seq.num_computed_tokens += 2
    assert runner.run_prefill(seq, 1) == 6
    seq.num_computed_tokens += 1

    assert model.past_lengths == [0, 2, 4]
    assert model.cache_length(seq._kv_cache) == seq.prompt_len
    assert seq.generated_token_ids == [6]

    assert runner.run_decode(seq) == 7
    assert model.past_lengths[-1] == seq.prompt_len
    assert model.cache_length(seq._kv_cache) == seq.num_tokens - 1


def test_prefill_finishes_request_when_max_tokens_is_one():
    config = tiny_engine_config()
    blocks = BlockManager(num_blocks=8, block_size=config.block_size)
    scheduler = Scheduler(config, blocks)
    runner = ModelRunner(TinyCacheModel(), tiny_model_config(), config, device="cpu")
    seq = Sequence([1, 2], SamplingParams(temperature=0, max_tokens=1))
    scheduler.add(seq)

    batch = scheduler.schedule()
    outputs = runner.run(batch)
    finished = scheduler.postprocess(batch, outputs)

    assert finished == [seq]
    assert seq.generated_token_ids == [3]
    assert seq.finish_reason == "length"
    assert seq.status == SequenceStatus.FINISHED
    assert not scheduler.has_unfinished()
    assert blocks.num_free_blocks == blocks.num_blocks


def test_prefill_output_updates_reserved_block_table_slot():
    config = tiny_engine_config()
    blocks = BlockManager(num_blocks=4, block_size=2)
    scheduler = Scheduler(config, blocks)
    runner = ModelRunner(
        TinyCacheModel(),
        tiny_model_config(),
        config,
        device="cpu",
        block_manager=blocks,
    )
    seq = Sequence([1, 2], SamplingParams(temperature=0, max_tokens=2))
    scheduler.add(seq)

    batch = scheduler.schedule()
    runner.run(batch)

    assert len(seq.block_table) == 2
    assert blocks.blocks[seq.block_table[1]].token_ids == [3]
    assert blocks.slot_mapping(seq, [2]) == [seq.block_table[1] * 2]


def test_decode_reserves_speculative_lookahead_blocks():
    config = tiny_engine_config(enable_spec_decode=True, max_spec_tokens=3)
    blocks = BlockManager(num_blocks=8, block_size=2)
    scheduler = Scheduler(config, blocks)
    seq = Sequence([1, 2], SamplingParams(temperature=0, max_tokens=8))
    blocks.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    seq.num_computed_tokens = seq.prompt_len
    seq.append_token(3)
    scheduler.running.append(seq)

    batch = scheduler.schedule()

    assert len(batch.items) == 1
    # 3 existing token positions + up to 4 emitted positions => 4 blocks.
    assert len(seq.block_table) == 4
    assert batch.items[0].num_tokens == 4


def test_chunked_prefill_rotates_long_prompt_behind_short_prompt():
    config = tiny_engine_config(
        enable_chunked_prefill=True,
        max_num_batched_tokens=4,
        max_prefill_chunk_size=2,
    )
    blocks = BlockManager(num_blocks=16, block_size=2)
    scheduler = Scheduler(config, blocks)
    long_seq = Sequence([1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=2))
    short_seq = Sequence([7, 8], SamplingParams(max_tokens=2))
    scheduler.add(long_seq)
    scheduler.add(short_seq)

    batch = scheduler.schedule()

    assert [(item.seq, item.num_tokens) for item in batch.items] == [
        (long_seq, 2),
        (short_seq, 2),
    ]
    assert list(scheduler.waiting) == [long_seq]
