"""OpenAI-compatible serving layer for the inference engine.

Wraps the engine in a FastAPI application with:
  POST /v1/completions       — text completions (streaming via SSE)
  POST /v1/chat/completions  — chat completions (prompt template)
  GET /health                 — engine readiness
  GET /metrics                — KV block usage, queue depths
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop: str | list[str] | None = None
    stream: bool = False
    logprobs: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False


class Choice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[Choice]


app = FastAPI()
engine = None
sampling_params_cls = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app(engine_instance, SamplingParams):
    global engine, sampling_params_cls
    engine = engine_instance
    sampling_params_cls = SamplingParams
    app.router.lifespan_context = lifespan
    return app


@app.get("/health")
def health():
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    return {"status": "ok", "device": engine.device}


@app.get("/metrics")
def metrics():
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    s = engine.scheduler.debug_state()
    return {
        "waiting_requests": s["waiting"],
        "running_requests": s["running"],
        "free_kv_blocks": s["blocks"]["free_blocks"],
        "used_kv_blocks": s["blocks"]["used_blocks"],
        "total_kv_blocks": s["blocks"]["total_blocks"],
        "engine_steps": engine.step_counter,
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    params = sampling_params_cls(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop,
        logprobs=req.logprobs,
    )

    if req.stream:
        return StreamingResponse(
            _stream_completions(req, params),
            media_type="text/event-stream",
        )

    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    outputs = engine.generate(prompts, params)

    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    choices = [
        Choice(index=i, text=out["generated_text"]) for i, out in enumerate(outputs)
    ]

    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=req.model,
        choices=choices,
    ).model_dump()


async def _stream_completions(req: CompletionRequest, params):
    request_id = engine.add_request(req.prompt, params)

    last_len = 0
    while engine.scheduler.has_unfinished():
        batch_finished = engine.step()

        for seq in batch_finished:
            if seq.request_id == request_id:
                prompt_text = (
                    req.prompt if isinstance(req.prompt, str) else req.prompt[0]
                )
                prompt_len = len(
                    engine.tokenizer.encode(prompt_text, add_special_tokens=True)
                )
                gen_text = engine.tokenizer.decode(
                    seq.token_ids[prompt_len:], skip_special_tokens=True
                )
                new_text = gen_text[last_len:]
                if new_text:
                    last_len = len(gen_text)
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "text": new_text,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if engine is None:
        raise HTTPException(503, "Engine not initialized")

    texts = [f"<|{msg.role}|>\n{msg.content}" for msg in req.messages]
    prompt = "\n".join(texts) + "\n<|assistant|>\n"

    comp_req = CompletionRequest(
        model=req.model,
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stream=req.stream,
    )
    return completions(comp_req)


def start_server(engine_instance, SamplingParams, host="0.0.0.0", port=8000):
    import uvicorn

    create_app(engine_instance, SamplingParams)
    uvicorn.run(app, host=host, port=port)
