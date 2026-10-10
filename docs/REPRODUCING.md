# Reproducing the project and measurements

Run from the repository root. Use one isolated environment per experiment and
one GPU process at a time. Timing varies with clocks, power, thermals, drivers
and background load; reproducing a protocol does not guarantee identical tok/s.

## 1. Install

For ordinary use, Python 3.10+ and an appropriate PyTorch build are required:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[paged,server,test]'
```

The October 6 serving experiment used Python **3.11.14**. With that Python in
a fresh environment, install the recorded direct dependencies instead:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements/serving.txt
python -m pip install -e '.[paged,server,test]'
```

The separate quantization experiment used Python **3.12.12** in another
environment:

```bash
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements/quantization.txt
python -m pip install -e '.[quantization,test]'
```

These files pin the direct versions recorded in the manifests, not a complete
transitive lockfile. Platform/wheel availability and driver compatibility still
matter. The records identify CUDA runtime 13.0 and driver 595.91.07 on SM120.
Check the installed runtime with `gpu_smoke`; do not substitute a CPU wheel and
interpret a skipped GPU suite as GPU validation. Other supported CUDA versions
may work but are a different measurement environment.

## 2. Checkpoint

```bash
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
sha256sum Qwen3-0.6B/config.json Qwen3-0.6B/model.safetensors
```

Recorded SHA-256:

| File | SHA-256 |
|---|---|
| `config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |

The original HF revision/tokenizer file hashes were not captured. If upstream
files change, the default download may differ: compare these checksums and
record your tokenizer/revision rather than claiming exact input reproduction.
Detailed artifacts retain actual prompt token IDs for auditing the input.
Load only trusted checkpoints: the HF integration enables remote model/tokenizer
code through `trust_remote_code=True`.

## 3. Correctness before speed

```bash
nvidia-smi
python -m nanovllm_pp.benchmarks.gpu_smoke
python -m ruff check nanovllm_pp tests examples
python -m pytest -q -o faulthandler_timeout=30
python -m nanovllm_pp.benchmarks.validate_paged_backend ./Qwen3-0.6B
python examples/generate.py ./Qwen3-0.6B
```

CPU tests do not download models. Optional CUDA/Triton tests skip if unavailable.
The model validator prints HF greedy parity, physical pool shape, prefix reuse
and final block state; tensor tolerance tests complement this small smoke test.
For CPU-only development, install `.[test,server]`; paged model execution is
CUDA-only. Restricting GPU devices or local socket access can prevent execution
even when dependencies are installed.

## 4. Repeated serving ablations

In the serving environment:

```bash
python -m nanovllm_pp.benchmarks.bench_runtime_evidence ./Qwen3-0.6B \
  --output-dir runs/runtime_evidence --repeats 3 --num-prompts 12 --seed 17
```

This runs 12 cases: batching widths 1/4/8, Poisson 4 requests/s, 64/256/512-token
prefix off/warm, 256-token cold start, and 72-page memory pressure. Config,
environment, checkpoint checksums, raw requests, iteration samples, per-case
JSON, `results.json` and `summary.csv` are saved. All cases use greedy forced
output lengths; speculation is off. Warmup and model load are excluded.

Use [RUNTIME_EVIDENCE.md](RUNTIME_EVIDENCE.md) to understand exactly what varies.
The published October artifacts came from a dirty worktree based on `fe112115`;
the improvements are now included in the repository. Metadata is not evidence
that pristine `fe112115` alone reproduces them. New results must preserve their
own revision/status and date instead of overwriting the recorded files.

## 5. Quantization memory/latency/fidelity

In the separate quantization environment:

```bash
python -m nanovllm_pp.benchmarks.bench_loading_evidence ./Qwen3-0.6B \
  --output runs/quantized_loading.json --repeats 3 --max-tokens 12
```

Each format is loaded alone and warmed before three generation trials. The
script prints outputs/metrics and records loaded/peak allocation, conversion,
token IDs, BF16 matches and the small NLL/logit probe. Custom storage prototypes
and NF4 are HF paths; quantization does not apply to the custom paged pool.
[Metric and arithmetic boundaries](QUANTIZATION_SCOPE.md).

## 6. Standard shapes and concurrency

```bash
python -m nanovllm_pp.benchmarks.bench_serving ./Qwen3-0.6B \
  --dataset-name random --num-prompts 16 --seed 17 \
  --random-input-len 256 --random-output-len 64 \
  --max-concurrency 8 --request-rate inf --kv-cache-num-blocks 512 \
  --max-model-len 512 --save-detailed --save-result runs/balanced.json
```

For decode-heavy use input/output 32/128; prefill-heavy uses 1024/16,
`--num-prompts 8 --max-concurrency 4 --max-model-len 2048`. Sweep concurrency
1/4/8. Use `--backend hf` for the reference, understanding that it is a different
model/dispatch implementation. `--request-rate 4` uses seeded Poisson arrivals;
`inf` is a burst, not a fixed offered-rate service guarantee.

Prefix ablation (repeat with `--no-enable-prefix-caching`):

```bash
python -m nanovllm_pp.benchmarks.bench_serving ./Qwen3-0.6B \
  --dataset-name prefix_repetition --num-prompts 8 --seed 17 \
  --prefix-len 256 --suffix-len 32 --prefix-output-len 32 \
  --max-concurrency 8 --kv-cache-num-blocks 512 --max-model-len 512 \
  --enable-prefix-caching --save-result runs/prefix_on.json
```

### ShareGPT dataset

This [dataset mirror](https://huggingface.co/datasets/learnanything/sharegpt_v3_unfiltered_cleaned_split/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)
publishes `ShareGPT_V3_unfiltered_cleaned_split.json` with the recorded SHA-256;
download separately and
observe its usage/license terms. The input is not redistributed here:

```bash
hf download learnanything/sharegpt_v3_unfiltered_cleaned_split \
  ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset --local-dir ./data
sha256sum data/ShareGPT_V3_unfiltered_cleaned_split.json
python -m nanovllm_pp.benchmarks.bench_serving ./Qwen3-0.6B \
  --dataset-name sharegpt --dataset-path ./data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 16 --seed 17 --max-concurrency 8 --max-output-len 128 \
  --kv-cache-num-blocks 512 --max-model-len 2048 \
  --save-detailed --save-result runs/sharegpt.json
```

Recorded dataset SHA-256:
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.
The adapter shuffles with seed 17, selects first-turn text pairs, and filters
to fit the context limit. For the historical arrival sweep use 12 prompts,
`--sharegpt-output-len 32`, and rates 1/4/inf. This is sampled conversation
traffic, not full multi-turn tool-use behavior or a production SLO study.

## 7. Outputs, traces and profiling

`bench_serving --save-detailed` captures request timing/token IDs and per-step
scheduler/page telemetry. The live demo modules in the root README print model
output plus a compact report; pass `--save-json runs/demo.json` to retain it.
`bench_paged_backend` is a whole-path HF/paged probe; the controlled harness is
the preferred one-variable batching/prefix study. Use `--help` for their flags.

`engine.enable_profiling()` enables host phase/request timing, not kernel-level
attribution. `engine.enable_tracing()` enables JSONL trace events; inspect its
signature before choosing a trace path. See `observability/` and
`benchmarks/bench_profiling.py`. Kernel attribution still requires a device
profiler. For JSONL export, enable tracing before generation, create `runs/`,
then call `engine._trace.write_jsonl("runs/trace.jsonl")`; `_trace` is an
internal interface, not a stable public schema. All new outputs belong under
ignored `runs/`.
