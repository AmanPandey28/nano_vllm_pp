"""Cancellation and failure paths must release physical-page ownership."""

import asyncio
from types import SimpleNamespace

import pytest

from conftest import TinyCacheModel, tiny_engine_config, tiny_model_config
from nanovllm_pp.engine.block_manager import BlockManager
from nanovllm_pp.engine.llm_engine import LLMEngine
from nanovllm_pp.engine.model_runner import ModelRunner
from nanovllm_pp.engine.scheduler import Scheduler
from nanovllm_pp.sampling_params import SamplingParams


def runtime():
    engine = LLMEngine.__new__(LLMEngine)
    engine.engine_config = tiny_engine_config()
    engine.block_manager = BlockManager(8, 2)
    engine.num_kv_blocks = 8
    engine.scheduler = Scheduler(engine.engine_config, engine.block_manager)
    engine.model_runner = ModelRunner(
        TinyCacheModel(),
        tiny_model_config(),
        engine.engine_config,
        "cpu",
        engine.block_manager,
    )
    engine.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [1, 2],
        decode=lambda ids, **kwargs: " ".join(str(token) for token in ids),
        eos_token_id=None,
    )
    engine.device = "cpu"
    engine.step_counter = 0
    engine._profiler = engine._request_tracker = engine._trace = None
    engine._sequences = {}
    return engine


def test_model_failure_reclaims_scheduled_requests_and_recovers():
    engine = runtime()
    request_id = engine.add_request("hello", SamplingParams(max_tokens=3))
    original = engine.model_runner.run

    def fail(batch):
        raise RuntimeError("injected model failure")

    engine.model_runner.run = fail
    with pytest.raises(RuntimeError, match="injected"):
        engine.step()
    assert engine.get_request(request_id).finish_reason == "error"
    assert not engine.scheduler.has_unfinished()
    engine.block_manager.assert_invariants([])
    engine.forget_request(request_id)
    engine.model_runner.run = original
    assert engine.generate("retry", SamplingParams(temperature=0, max_tokens=1))
    engine.block_manager.assert_invariants([])


def test_abort_waiting_and_running_requests_is_idempotent():
    engine = runtime().enable_profiling()
    request_id = engine.add_request("waiting", SamplingParams(max_tokens=3))
    engine.abort_request(request_id)
    engine.abort_request(request_id)
    engine.forget_request(request_id)
    request_id = engine.add_request("running", SamplingParams(max_tokens=3))
    engine.step()
    engine.abort_request(request_id)
    engine.forget_request(request_id)
    engine.block_manager.assert_invariants([])
    assert not engine.scheduler.has_unfinished()


def test_invalid_multi_prompt_http_submission_does_not_leave_partial_requests():
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from nanovllm_pp.server.app import CompletionRequest, create_app

    engine = runtime()
    engine.tokenizer.encode = lambda text, **kwargs: (
        [1] * (200 if text == "invalid" else 2)
    )
    api = create_app(engine, SamplingParams)
    endpoint = next(
        route.endpoint for route in api.routes if route.path == "/v1/completions"
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            endpoint(CompletionRequest(model="tiny", prompt=["valid", "invalid"]))
        )
    assert error.value.status_code == 400
    assert not engine.scheduler.has_unfinished()
    assert engine._sequences == {}


def test_closing_stream_reclaims_unfinished_request():
    pytest.importorskip("fastapi")
    from nanovllm_pp.server.app import CompletionRequest, create_app

    engine = runtime()
    api = create_app(engine, SamplingParams)
    endpoint = next(
        route.endpoint for route in api.routes if route.path == "/v1/completions"
    )

    async def consume_then_close():
        response = await endpoint(
            CompletionRequest(
                model="tiny", prompt="hello", max_tokens=8, temperature=0, stream=True
            )
        )
        chunk = await asyncio.wait_for(anext(response.body_iterator), timeout=3)
        assert "data:" in chunk
        assert engine.scheduler.has_unfinished()
        await asyncio.wait_for(response.body_iterator.aclose(), timeout=3)

    asyncio.run(consume_then_close())
    assert not engine.scheduler.has_unfinished()
    assert engine._sequences == {}
    engine.block_manager.assert_invariants([])
