# RTX 5050 GPU validation

This is a retained environment manifest and smoke/performance record for one
laptop. It is evidence for these exact runs, not a universal performance claim.

## Why CUDA was initially invisible

The restricted automation process could see the PCI device but not the NVIDIA
character devices under `/dev/nvidia*`; `nvidia-smi` therefore failed and
`torch.cuda.is_available()` returned `False`. Running the same commands with
host device access exposed CUDA immediately. No driver or PyTorch reinstall was
needed.

Use this fail-fast sequence before a GPU run:

```bash
nvidia-smi
python -m nanovllm_pp.benchmarks.gpu_smoke
```

Performance scripts request `cuda` explicitly. If device access disappears,
they raise instead of silently producing CPU numbers. This follows PyTorch's
[device-agnostic CUDA guidance](https://docs.pytorch.org/docs/main/notes/cuda.html)
and synchronizes timed regions because CUDA execution is asynchronous.

## Environment captured on August 5, 2026

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| Compute capability | 12.0 |
| NVIDIA driver | 595.84 |
| Driver-reported CUDA compatibility | 13.2 |
| NVIDIA-SMI memory | 8,151 MiB |
| PyTorch | 2.12.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| Transformers | 5.11.0 |
| bitsandbytes | 0.49.2 |
| Automatic compute dtype | BF16 (supported) |
| Model | Local Qwen3-0.6B, 28 layers, BF16 checkpoint |
| `config.json` SHA-256 | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| `model.safetensors` SHA-256 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |

The smoke test completed a 1024-by-1024 BF16 matrix multiplication on
`cuda:0`. After the block-table correctness revision, the host-visible test
suite ran all 42 tests, including CUDA INT8/INT4 execution tests. In the
restricted process, 40 passed and the two CUDA tests skipped as designed.

## Physical paged-KV validation on August 6, 2026

The new `model_backend="paged"` path was validated separately from the older
HF-cache measurements below. The complete suite passed with host GPU access:

```text
52 passed in 1.54s
```

The CUDA test writes and reads deliberately non-contiguous physical pages,
compiles the Triton online-softmax decode kernel for SM120, and compares its
GQA output with a PyTorch block-table oracle. A separate tiny randomly
initialized Qwen3 test copies identical weights into the HF and custom models
and checks both prefill and decode logits. All passed.

The repeatable end-to-end command is:

```bash
python -m nanovllm_pp.benchmarks.validate_paged_backend \
  /home/aman/dev/Projects/nano_vllm_++/Qwen3-0.6B
```

With 64 four-token pages it allocated a physical tensor of shape
`[2, 28, 64, 4, 8, 128]` (29,360,128 bytes). Both four-token greedy samples
matched the HF backend. A second 13-token prompt sharing a common prefix reused
12 physical K/V tokens and computed only the final prompt token; the observed
block table was `[6, 7, 8, 10]`. A repetitive speculative workload proposed
five tokens, accepted all five, and emitted seven decode tokens in two target
forwards after the initial prefill token.

Automatic capacity sizing was also exercised without a block-count override.
At the default 256-token page size and 0.90 utilization it allocated 188 pages:
`[2, 28, 188, 256, 8, 128]`, or 5.140625 GiB of actual BF16 K/V storage, then
completed generation. This is now a physical allocation rather than the older
logical admission estimate.

Greedy identity is a sensitive smoke test, not a universal numerical contract.
In a longer six-token sample, one of two prompts chose a different sixth token
after five identical tokens; custom Triton/PyTorch reduction order can change
an argmax when logits are nearly tied. The architectural tests therefore also
compare tensors within tolerance, and performance/quality claims should use a
defined evaluation corpus rather than require bitwise equality.

The post-revision Qwen3 engine tutorial also completed on `cuda:0`: three
prompts generated 24 tokens each in 24 engine steps. After termination the
scheduler reported empty waiting/running queues and all 188 blocks returned to
the free pool. That run validates the integrated block-table metadata and
reclamation path; its sampled 1.53 s latency is not treated as a benchmark
distribution.

The configured FastAPI server was also exercised against the same model and
device. Health, non-streaming completions with log probabilities, SSE
streaming, chat completions, and `/metrics` all returned successfully. After
three completed requests, the metrics snapshot reported zero queued/running
requests, all 188 logical blocks free, and populated TTFT, TPOT, E2E, and
engine-phase summaries.

### Current HF-versus-paged execution probe

`bench_paged_backend.py` compares the current backends on the same engine API.
The recorded run used three prompts, 16 forced output tokens per prompt, 128
physical pages of 16 tokens, and the median of three warmed, CUDA-synchronized
trials:

| Backend | Time | Output rate | TTFT p50 | TPOT p50 | E2E p50 | Resident | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| HF reference | 0.642 s | 74.8 tok/s | 41.9 ms | 40.0 ms | 641.3 ms | 1.14 GiB | 1.15 GiB |
| Physical paged | 0.251 s | 191.4 tok/s | 27.4 ms | 14.9 ms | 250.4 ms | 1.36 GiB | 1.36 GiB |

The observed output-throughput ratio was 2.56x. This is primarily a comparison
of execution architectures: the HF reference dispatches requests serially,
whereas the paged runner batches the three same-mode requests and uses direct
paged decode. It does **not** isolate page-table indirection as the cause of the
speedup. Decode remained the dominant phase; the median paged run averaged
13.899 ms in decode and 15.625 ms total per engine step.

The paged allocation was `[2, 28, 128, 16, 8, 128]`, or 234,881,024 bytes.
Two of three 16-token completions matched the HF reference exactly. The third
encountered a numerically sensitive argmax difference, consistent with the
longer parity caveat above. Reproduce the run with:

```bash
python -m nanovllm_pp.benchmarks.bench_paged_backend \
  /home/aman/dev/Projects/nano_vllm_++/Qwen3-0.6B
```

## Observed synchronized results

All timed paths perform warm-up and synchronize CUDA before and after the
measured region. Model load time is excluded. The laptop's power/thermal state
was not locked, so small differences should not be overinterpreted.

### Greedy execution comparison

Three prompts, 16 output tokens each:

| Path | Median-like single synchronized run | Output rate | Correctness |
|---|---:|---:|---|
| Full-context recomputation | 0.646 s | 74.3 tok/s | Reference |
| Hugging Face K/V cache | 0.641 s | 74.9 tok/s | Exact token parity |
| NanoVLLM++ engine | 0.675 s | 71.1 tok/s | Exact token parity |

At these short sequence lengths, cache reuse does not amortize enough work to
produce a meaningful speedup, and the engine serializes requests. This is a
useful negative result rather than evidence against K/V caching at long context.

### Chunked prefill

One long and five short prompts, 12 output tokens each, actual chunk size 64:

| Path | Time | Steps | Rate | Long TTFT | Short mean TTFT |
|---|---:|---:|---:|---:|---:|
| No chunking | 0.944 s | 12 | 76.3 tok/s | 84.6 ms | 84.5 ms |
| Chunked | 1.030 s | 13 | 69.9 tok/s | 191.9 ms | 87.5 ms |

Generated tokens matched exactly. This workload did not benefit from chunking:
the runner performs serial model calls, the prompts arrived together, and the
extra chunk added a step. The benchmark now genuinely crosses a chunk boundary;
the former 512-token setting did not.

### Greedy n-gram speculative decoding

One deliberately repetitive prompt, 12 output tokens, three synchronized
samples:

| Path | Median | Target forwards | Accepted/proposed | Correctness |
|---|---:|---:|---:|---|
| Cached greedy | 0.170 s | 12 | N/A | Reference |
| N-gram speculation | 0.063 s | 4 | 8/8 | Exact token parity |

This is about 2.7x for the selected high-acceptance prompt. It must not be
generalized to arbitrary text, batches, proposal lengths, or acceptance rates.

### bitsandbytes storage and execution

Three prompts, 12 output tokens, median of three synchronized samples:

| Format | Allocated model memory | Time | Rate | Exact BF16 token matches |
|---|---:|---:|---:|---:|
| BF16 | 1.11 GiB | 0.474 s | 76.0 tok/s | 3/3 |
| NF4 | 0.53 GiB | 0.953 s | 37.8 tok/s | 0/3 |
| INT8 | 0.76 GiB | 2.742 s | 13.1 tok/s | 2/3 |

NF4 saved about 52% of allocated model memory but was slower for this small
model/workload. Exact token matching is only a sensitive smoke signal, not a
replacement for perplexity or task-level accuracy evaluation.

The educational per-channel INT8 module reduced counted model storage from
1,136.9 MiB to 718.2 MiB on the same checkpoint, retained exact tokens for the
single smoke prompt, and slowed the full-context path from 0.112 s to 0.372 s.
That slowdown is expected because it reconstructs full floating weights before
every matmul; it is not a fused INT8 compute kernel.

## Remaining hardware work

- Repeat under a fixed power/performance mode with more trials and confidence
  intervals.
- Sweep prompt/output length, concurrency, chunk size, and speculative
  acceptance rather than relying on one small workload.
- Run long-context memory stress and out-of-memory recovery tests.
- Evaluate quantized perplexity/task accuracy on a representative dataset.
- Capture Nsight Systems/Compute traces and correlate host gaps, kernels,
  memory bandwidth, and power.
- Replace gathered paged prefill with a fused variable-length kernel and add
  graph capture before making serving-scale throughput claims.
