# Usage and API boundaries

Install and download the checkpoint using [REPRODUCING.md](REPRODUCING.md).
Always use `--help` to inspect executable flags. No weights are bundled.

## Python interface

`LLM` inherits `LLMEngine`; `generate(prompt_or_list, SamplingParams(...))`
returns a list, even for one prompt. Each output includes `generated_text`,
`text` (prompt + output), `token_ids`, prompt/completion counts and finish reason.
See [examples/generate.py](../examples/generate.py).

For control over arrivals:

```python
rid = llm.add_request("A KV cache stores", SamplingParams(temperature=0, max_tokens=16))
while not llm.is_request_finished(rid):
    llm.step()                    # executes scheduled work for ALL live requests
output = llm.get_output(rid)
llm.forget_request(rid)           # releases retained result, not just KV pages
```

`abort_request(rid)` cancels waiting, partially prefilled or running work at an
iteration boundary and releases its page references. Call `forget_request`
after inspecting the cancelled output. `generate` handles finished-result
cleanup automatically. The engine is not thread-safe; external concurrent
callers must serialize admission, stepping, cancellation and snapshots.

`SamplingParams` supports temperature, top-k/top-p/min-p, max tokens, stop
strings/IDs, EOS ignoring, seed and logprobs. Greedy mode is temperature zero.
Stop strings and stochastic sampling disable n-gram speculation for the request.
Benchmark `ignore_eos=True` forces work; normal examples honor EOS.

## Server launcher

```bash
python -m nanovllm_pp.server ./Qwen3-0.6B \
  --backend paged --device cuda --dtype bfloat16 \
  --block-size 16 --kv-cache-num-blocks 256 \
  --max-num-seqs 4 --max-num-batched-tokens 256 --max-model-len 1024 \
  --max-prefill-chunk-size 128 --enable-prefix-caching
python -m nanovllm_pp.server --help
```

Chunking is enabled by default in this launcher; disable it with
`--no-enable-chunked-prefill`. Prefix reuse and speculation are opt-in via
`--enable-prefix-caching` and `--enable-spec-decode`. Library defaults differ:
`EngineConfig` defaults to HF, with chunking/prefix/speculation disabled.

Use `--backend hf` for a compatible HF causal LM. Optional
`--load-in-4bit` (NF4) and `--load-in-8bit` require the quantization extra and
CUDA; they are mutually exclusive and rejected on paged. HF cannot physically
share prefixes. A block override fixes the physical pool size on paged; on HF
it bounds admission metadata rather than allocating those tensor pages.

Run exactly one server process/model; multiple worker processes would load
independent models and pools. Do not run the unconfigured module-level FastAPI
`app` directly. `create_app(engine, SamplingParams)` is the embedding interface;
`start_server(...)` constructs it. Bind defaults to `127.0.0.1:8000`; no auth,
rate limiting, TLS or multi-tenant guarantees are provided.

## Endpoints

| Endpoint | Contract |
|---|---|
| `GET /health` | Liveness status and device; not a quality check |
| `GET /metrics` | JSON engine/request counters and latency summaries, not Prometheus exposition |
| `POST /v1/completions` | `model`, `prompt` string/list, sampling fields, optional SSE |
| `POST /v1/chat/completions` | `model`, `messages` with role/content; tokenizer chat template |
| `GET /docs` | FastAPI-generated field schema and interactive API |

Completion fields: `model`, `prompt`, `max_tokens`, `temperature`, `top_p`,
`top_k`, `min_p`, `stop`, `stream`, `logprobs`, `seed`. Chat fields: `model`,
`messages`, `max_tokens`, `temperature`, `top_p`, `stop`, `stream`, `seed`.
`model` is required but is only echoed; it does not select another checkpoint.
Prompt length + max output must fit `max_model_len`.

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","prompt":["The capital of France is","A transformer is"],"max_tokens":16,"temperature":0}'
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Explain a KV cache in one sentence."}],"max_tokens":64,"temperature":0}'
```

For streaming, use a single prompt, `"stream":true` and `curl -N`. SSE emits
text deltas, a finish-reason chunk and `[DONE]`. Chat streaming currently uses
completion-shaped chunks, not standard chat `delta` chunks. Unsupported OpenAI
features (tool calls, multimodal inputs, full response schemas) are not implied.
Qwen3's default chat template may emit reasoning/`<think>` text; this server
does not expose a thinking-mode switch.

The app lock protects model calls and request ownership while allowing new
requests between steps. Invalid multi-prompt submissions are rolled back;
stream closure and model errors release affected requests. Cancellation waits
for an in-flight model thread before releasing the lock. A fatal CUDA context
error may still require restarting the process.

## Metric definitions

The server timestamps **engine enqueue after tokenization**, first computed
output, and engine completion. It excludes network/serialization/first-byte
delivery. The direct harness starts immediately before `add_request`, includes
tokenization and the engine queue, and observes outputs after CUDA-synchronized
steps. Neither is an HTTP load test.

| Metric | Definition |
|---|---|
| TTFT | First observed token minus request start |
| Request TPOT | Post-first-token duration / `(output_tokens - 1)`; omitted for one token |
| ITL | Consecutive token-observation differences; harness pools intervals |
| E2E | Completion minus request start |
| Client wait | Scheduled arrival to actual send, before the harness concurrency gate |
| Output throughput | Generated output tokens / first-send-to-last-completion duration |
| Total-token throughput | `(input + output tokens) / duration`, not model work actually executed |
| Processed-token throughput | `(uncached input + output - requests) / duration` for ordinary decode |
| KV utilization | Pages with live references / pool pages; zero-reference cached pages are reclaimable |
| Prefix token hit fraction | Physically reused prompt tokens / admitted prompt tokens |
| Prefix request hit fraction | Requests reusing at least one full page / admitted requests |

The final emitted token is not forwarded into KV until another decode step.
Server TPOT uses completion as the last-token boundary; harness TPOT uses the
last observed token. Server percentiles are nearest-rank; harness percentiles
are interpolated. Prefix metrics accumulate until reset; warmup is excluded in
benchmarks. Speculative bursts share an observation timestamp, not independent
streaming ITLs. Host phase timings are not CUDA kernel timings. Goodput requires
explicit SLOs and is not measured here.
