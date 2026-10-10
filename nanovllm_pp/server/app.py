"""Small OpenAI-style FastAPI surface for the educational engine.

The app factory keeps engine state in closures instead of module globals.  A
single asyncio lock protects the non-thread-safe scheduler/model runner, while
each request yields between engine steps so concurrent clients can be admitted
to the same scheduler iteration.
"""

import asyncio
import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from anyio import CancelScope

from ..observability.profiler import serving_metrics_snapshot


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    stop: str | list[str] | None = None
    stream: bool = False
    logprobs: int | None = None
    seed: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None


class Choice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = None
    logprobs: list | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[Choice]
    usage: dict[str, int]


# Kept for ``from nanovllm_pp.server.app import app`` compatibility.  The
# configured application is returned by create_app/start_server.
app = FastAPI(title="NanoVLLM++ (unconfigured)")


def create_app(engine, SamplingParams):
    api = FastAPI(title="NanoVLLM++")
    step_lock = asyncio.Lock()
    if getattr(engine, "_request_tracker", None) is None:
        engine.enable_profiling()

    def make_params(req):
        return SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            top_k=getattr(req, "top_k", 0),
            min_p=getattr(req, "min_p", 0.0),
            stop=req.stop,
            logprobs=getattr(req, "logprobs", None),
            seed=req.seed,
        )

    async def add_requests(prompts, params):
        request_ids = []
        try:
            async with step_lock:
                try:
                    for prompt in prompts:
                        request_ids.append(engine.add_request(prompt, params))
                except (TypeError, ValueError, NotImplementedError):
                    for rid in request_ids:
                        engine.abort_request(rid)
                        engine.forget_request(rid)
                    raise
                return request_ids
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def step_once():
        try:
            async with step_lock:
                worker = asyncio.create_task(asyncio.to_thread(engine.step))
                try:
                    return await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # Cancellation must not release the lock while the model
                    # thread is still mutating scheduler/KV state.
                    with CancelScope(shield=True):
                        await worker
                    raise
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def all_finished(request_ids):
        async with step_lock:
            return all(engine.is_request_finished(rid) for rid in request_ids)

    async def snapshot_outputs(request_ids, *, forget=False):
        async with step_lock:
            outputs = [engine.get_output(rid) for rid in request_ids]
            if forget:
                for rid in request_ids:
                    engine.forget_request(rid)
            return outputs

    async def drive(request_ids):
        try:
            while not await all_finished(request_ids):
                await step_once()
                await asyncio.sleep(0)
            return await snapshot_outputs(request_ids, forget=True)
        finally:
            await release_requests(request_ids)

    async def release_requests(request_ids):
        # Starlette uses level-triggered AnyIO cancellation on disconnect;
        # shield the cleanup so its lock acquisition cannot be cancelled again.
        with CancelScope(shield=True):
            async with step_lock:
                for rid in request_ids:
                    engine.abort_request(rid)
                    engine.forget_request(rid)

    @api.get("/health")
    async def health():
        return {"status": "ok", "device": engine.device}

    @api.get("/metrics")
    async def metrics():
        async with step_lock:
            return serving_metrics_snapshot(engine)

    @api.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if req.stream and not isinstance(req.prompt, str):
            raise HTTPException(400, "streaming accepts one prompt per request")
        try:
            params = make_params(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if req.stream:
            return StreamingResponse(
                stream_completion(req, params), media_type="text/event-stream"
            )

        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        request_ids = await add_requests(prompts, params)
        outputs = await drive(request_ids)
        choices = [
            Choice(
                index=index,
                text=output["generated_text"],
                finish_reason=output["finish_reason"],
                logprobs=output["logprobs"] if req.logprobs else None,
            )
            for index, output in enumerate(outputs)
        ]
        usage = {
            "prompt_tokens": sum(o["num_prompt_tokens"] for o in outputs),
            "completion_tokens": sum(o["num_completion_tokens"] for o in outputs),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=choices,
            usage=usage,
        ).model_dump()

    async def stream_completion(req: CompletionRequest, params):
        request_id = (await add_requests([req.prompt], params))[0]
        try:
            async for chunk in stream_tokens(req, request_id):
                yield chunk
        finally:
            await release_requests([request_id])

    async def stream_tokens(req, request_id):
        previous = ""
        while not await all_finished([request_id]):
            await step_once()
            output = (await snapshot_outputs([request_id]))[0]
            current = output["generated_text"]
            if current.startswith(previous):
                delta = current[len(previous) :]
            else:
                # Decoding cleanup can very occasionally rewrite whitespace.
                # Sending the current snapshot is safer than silently dropping it.
                delta = current
            if delta:
                previous = current
                chunk = {
                    "id": request_id,
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {"index": 0, "text": delta, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)

        output = (await snapshot_outputs([request_id], forget=True))[0]
        final_chunk = {
            "id": request_id,
            "object": "text_completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": output["finish_reason"],
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    def format_chat(messages):
        values = [message.model_dump() for message in messages]
        if hasattr(engine.tokenizer, "apply_chat_template"):
            try:
                return engine.tokenizer.apply_chat_template(
                    values, tokenize=False, add_generation_prompt=True
                )
            except (ValueError, TypeError):
                pass
        return "\n".join(
            f"<|{message.role}|>\n{message.content}" for message in messages
        ) + "\n<|assistant|>\n"

    @api.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        completion_request = CompletionRequest(
            model=req.model,
            prompt=format_chat(req.messages),
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
            stream=req.stream,
            seed=req.seed,
        )
        response = await completions(completion_request)
        # Streaming already uses the standard SSE response.  For non-streaming,
        # reshape text choices to the OpenAI chat message form.
        if isinstance(response, StreamingResponse):
            return response
        response["object"] = "chat.completion"
        for choice in response["choices"]:
            choice["message"] = {
                "role": "assistant",
                "content": choice.pop("text"),
            }
        return response

    return api


def start_server(engine_instance, SamplingParams, host="0.0.0.0", port=8000):
    import uvicorn

    configured_app = create_app(engine_instance, SamplingParams)
    uvicorn.run(configured_app, host=host, port=port)
