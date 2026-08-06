import pytest
import torch

from nanovllm_pp.config import EngineConfig
from nanovllm_pp.device import resolve_device, resolve_dtype
from nanovllm_pp.engine.llm_engine import LLMEngine
from nanovllm_pp.engine.model_runner import Sampler
from nanovllm_pp.sampling_params import SamplingParams


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_size": 0}, "block_size"),
        ({"gpu_memory_utilization": 1.1}, "gpu_memory_utilization"),
        ({"max_spec_tokens": 0}, "max_spec_tokens"),
        ({"spec_proposer_type": "draft_model"}, "ngram"),
        ({"device": ""}, "device"),
        ({"dtype": "fp8"}, "dtype"),
        ({"model_backend": "unknown"}, "model_backend"),
        ({"kv_cache_num_blocks": 0}, "kv_cache_num_blocks"),
    ],
)
def test_engine_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EngineConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"max_tokens": 0},
        {"top_k": -1},
        {"top_p": 0.0},
        {"min_p": 1.1},
        {"logprobs": 0},
        {"seed": -1},
        {"stop": ""},
    ],
)
def test_sampling_params_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_sampler_reports_raw_logprob_and_is_seed_reproducible():
    sampler = Sampler(vocab_size=4)
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    params = SamplingParams(temperature=0.8, max_tokens=1, seed=17, logprobs=2)
    first = torch.Generator().manual_seed(17)
    second = torch.Generator().manual_seed(17)

    token_a, info_a = sampler(logits, params, first)
    token_b, info_b = sampler(logits, params, second)

    assert token_a == token_b
    assert info_a == info_b
    assert token_a in info_a["top_logprobs"] or len(info_a["top_logprobs"]) == 2
    expected = torch.log_softmax(logits, dim=-1)[token_a].item()
    assert info_a["logprob"] == pytest.approx(expected)


def test_explicit_cuda_request_fails_instead_of_silently_using_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_engine_checks_explicit_cuda_before_loading_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        LLMEngine("model-must-not-be-loaded", device="cuda")


def test_cpu_auto_dtype_is_float32():
    assert resolve_dtype(torch.device("cpu"), "auto") == torch.float32
