from conftest import TinyCacheModel, tiny_engine_config, tiny_model_config

from nanovllm_pp.engine.model_runner import ModelRunner
from nanovllm_pp.engine.sequence import Sequence
from nanovllm_pp.sampling_params import SamplingParams
from nanovllm_pp.spec_decode.manager import SpecDecodeManager


def make_ready_sequence(model, *, eos_token_id=None):
    seq = Sequence(
        [1, 2, 3, 1, 2], SamplingParams(temperature=0, max_tokens=8)
    )
    seq.eos_token_id = eos_token_id
    # The normal decoder invariant: cache every sequence token except the last.
    prefix = model.forward(
        __import__("torch").tensor([[1, 2, 3, 1]]), use_cache=True
    )
    seq._kv_cache = prefix.past_key_values
    seq.num_computed_tokens = seq.prompt_len
    return seq


def make_manager(transitions):
    model = TinyCacheModel(transitions=transitions)
    runner = ModelRunner(
        model, tiny_model_config(), tiny_engine_config(), device="cpu"
    )
    return model, SpecDecodeManager(runner, max_spec_tokens=2)


def test_rejection_accepts_prefix_emits_correction_and_crops_cache():
    model, manager = make_manager({2: 3, 3: 4})
    seq = make_ready_sequence(model)

    emitted = manager.step(seq)

    assert emitted == [3, 4]
    assert seq.generated_token_ids == [3, 4]
    assert manager.metrics.tokens_proposed == 2
    assert manager.metrics.tokens_accepted == 1
    assert model.cache_length(seq._kv_cache) == seq.num_tokens - 1


def test_full_acceptance_adds_bonus_and_matches_serial_greedy():
    transitions = {2: 3, 3: 1, 1: 9}
    model, manager = make_manager(transitions)
    speculative = make_ready_sequence(model)
    serial = make_ready_sequence(model)

    emitted = manager.step(speculative)
    for _ in range(3):
        manager.model_runner.run_decode(serial)

    assert emitted == [3, 1, 9]
    assert speculative.generated_token_ids == serial.generated_token_ids
    assert model.cache_length(speculative._kv_cache) == speculative.num_tokens - 1
    assert manager.metrics.tokens_accepted == 2


def test_eos_inside_speculative_burst_truncates_output_and_cache():
    model, manager = make_manager({2: 3, 3: 1, 1: 9})
    seq = make_ready_sequence(model, eos_token_id=1)

    emitted = manager.step(seq)

    assert emitted == [3, 1]
    assert seq.last_token == 1
    assert model.cache_length(seq._kv_cache) == seq.num_tokens - 1


def test_stop_string_forces_single_token_steps():
    model, manager = make_manager({2: 3})
    seq = make_ready_sequence(model)
    seq.sampling_params.stop = ["boundary"]

    emitted = manager.step(seq)

    assert emitted == [3]
    assert manager.metrics.tokens_proposed == 0
