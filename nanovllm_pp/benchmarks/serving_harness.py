"""Engine-level serving harness with arrival, latency, and cache measurements."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field

import torch

from nanovllm_pp import SamplingParams
from nanovllm_pp.device import cuda_device_info, synchronize

from .workloads import BenchmarkWorkload


@dataclass
class RequestRecord:
    index: int
    source: str
    scheduled_time: float
    send_time: float
    prompt_tokens: int
    requested_output_tokens: int
    request_id: str
    cached_prompt_tokens: int = 0
    token_times: list[float] = field(default_factory=list)
    completion_time: float | None = None
    generated_token_ids: list[int] = field(default_factory=list)


def percentile(values, percent: float) -> float:
    """Linearly interpolated percentile for a non-empty numeric sequence."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percent / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def latency_summary(values_ms) -> dict:
    if not values_ms:
        return {}
    return {
        "mean": round(sum(values_ms) / len(values_ms), 3),
        "p50": round(percentile(values_ms, 50), 3),
        "p95": round(percentile(values_ms, 95), 3),
        "p99": round(percentile(values_ms, 99), 3),
        "min": round(min(values_ms), 3),
        "max": round(max(values_ms), 3),
    }


def request_metrics(records: list[RequestRecord], duration_s: float) -> dict:
    """Aggregate vLLM-style serving metrics from completed request records."""
    if not records or duration_s <= 0:
        raise ValueError("completed records and positive duration are required")
    ttft, tpot, itl, e2e, client_wait = [], [], [], [], []
    total_input = 0
    total_output = 0
    total_cached = 0
    for record in records:
        if record.completion_time is None or not record.token_times:
            raise ValueError("every benchmark request must complete and emit a token")
        total_input += record.prompt_tokens
        total_output += len(record.token_times)
        total_cached += record.cached_prompt_tokens
        ttft.append((record.token_times[0] - record.send_time) * 1000)
        e2e.append((record.completion_time - record.send_time) * 1000)
        client_wait.append((record.send_time - record.scheduled_time) * 1000)
        intervals = [
            (later - earlier) * 1000
            for earlier, later in zip(record.token_times, record.token_times[1:])
        ]
        itl.extend(intervals)
        if intervals:
            tpot.append(sum(intervals) / len(intervals))

    return {
        "successful_requests": len(records),
        "benchmark_duration_s": round(duration_s, 6),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "request_throughput_rps": round(len(records) / duration_s, 4),
        "input_token_throughput_tps": round(total_input / duration_s, 3),
        "output_token_throughput_tps": round(total_output / duration_s, 3),
        "total_token_throughput_tps": round(
            (total_input + total_output) / duration_s, 3
        ),
        "cached_prompt_tokens": total_cached,
        "prefix_cache_hit_fraction": round(
            total_cached / total_input if total_input else 0.0, 6
        ),
        "latency_ms": {
            "ttft": latency_summary(ttft),
            "tpot": latency_summary(tpot),
            "itl": latency_summary(itl),
            "e2e": latency_summary(e2e),
            "client_wait": latency_summary(client_wait),
        },
    }


def poisson_arrival_offsets(num_requests: int, request_rate, seed: int) -> list[float]:
    if num_requests <= 0:
        return []
    if math.isinf(request_rate):
        return [0.0] * num_requests
    if request_rate <= 0:
        raise ValueError("request_rate must be positive or infinity")
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, num_requests):
        offsets.append(offsets[-1] + rng.expovariate(request_rate))
    return offsets


def _warm_engine(engine, workload: BenchmarkWorkload) -> None:
    warmups = workload.warmup_requests
    if not warmups:
        # Triton decode specializes on a power-of-two maximum context. Warm one
        # representative for every bucket used by this workload so JIT compile
        # time never appears as an extreme measured ITL sample.
        representatives = {}
        for request in workload.requests:
            context = (
                len(engine.tokenizer.encode(request.prompt, add_special_tokens=True))
                + request.output_tokens
            )
            bucket = max(16, 1 << max(0, context - 1).bit_length())
            representatives.setdefault(bucket, request)
        warmups = tuple(representatives.values())
    params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=request.output_tokens,
            ignore_eos=True,
        )
        for request in warmups
    ]
    engine.generate([request.prompt for request in warmups], params)


def run_serving_workload(
    engine,
    workload: BenchmarkWorkload,
    *,
    request_rate: float,
    max_concurrency: int,
    seed: int,
    save_detailed: bool = False,
) -> dict:
    """Run a closed-concurrency workload with optional Poisson arrivals.

    Requests become eligible according to ``request_rate``. At most
    ``max_concurrency`` requests are sent to the engine at once; a request's
    latency clock starts when that client slot becomes available, while
    ``client_wait`` reports time spent behind the benchmark-side semaphore.
    """
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if engine.scheduler.has_unfinished():
        raise RuntimeError("benchmark requires an idle engine")

    _warm_engine(engine, workload)
    engine.enable_profiling()
    device = torch.device(engine.device)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        resident_bytes = torch.cuda.memory_allocated(device)
    else:
        resident_bytes = 0

    offsets = poisson_arrival_offsets(len(workload.requests), request_rate, seed)
    records_by_id: dict[str, RequestRecord] = {}
    completed: list[RequestRecord] = []
    active_ids: set[str] = set()
    next_request = 0
    maximum_concurrency = 0
    maximum_pages_used = 0
    benchmark_start = time.perf_counter()

    while next_request < len(workload.requests) or active_ids:
        now = time.perf_counter()
        elapsed = now - benchmark_start
        while (
            next_request < len(workload.requests)
            and offsets[next_request] <= elapsed
            and len(active_ids) < max_concurrency
        ):
            request = workload.requests[next_request]
            send_time = time.perf_counter()
            params = SamplingParams(
                temperature=0.0,
                max_tokens=request.output_tokens,
                ignore_eos=True,
            )
            request_id = engine.add_request(request.prompt, params)
            sequence = engine.get_request(request_id)
            records_by_id[request_id] = RequestRecord(
                index=next_request,
                source=request.source,
                scheduled_time=benchmark_start + offsets[next_request],
                send_time=send_time,
                prompt_tokens=sequence.prompt_len,
                requested_output_tokens=request.output_tokens,
                request_id=request_id,
            )
            active_ids.add(request_id)
            next_request += 1
            maximum_concurrency = max(maximum_concurrency, len(active_ids))

        if engine.scheduler.has_unfinished():
            engine.step()
            synchronize(device)
            observed_time = time.perf_counter()
            finished_ids = []
            for request_id in active_ids:
                sequence = engine.get_request(request_id)
                record = records_by_id[request_id]
                record.cached_prompt_tokens = max(
                    record.cached_prompt_tokens, sequence.num_cached_tokens
                )
                new_tokens = sequence.num_generated_tokens - len(record.token_times)
                if new_tokens > 0:
                    # Ordinary decode emits one token per step. Speculative
                    # bursts share an observation timestamp because the public
                    # engine returns them atomically.
                    record.token_times.extend([observed_time] * new_tokens)
                if engine.is_request_finished(request_id):
                    record.completion_time = observed_time
                    record.generated_token_ids = list(sequence.generated_token_ids)
                    finished_ids.append(request_id)
            for request_id in finished_ids:
                record = records_by_id[request_id]
                completed.append(record)
                active_ids.remove(request_id)
                engine.forget_request(request_id)
            maximum_pages_used = max(
                maximum_pages_used, engine.block_manager.stats()["used_blocks"]
            )
            continue

        if next_request < len(workload.requests):
            wake_time = benchmark_start + offsets[next_request]
            delay = wake_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            continue
        if active_ids:
            raise RuntimeError("active requests remain but the engine has no work")

    synchronize(device)
    benchmark_end = max(record.completion_time for record in completed)
    first_send = min(record.send_time for record in completed)
    duration = benchmark_end - first_send
    completed.sort(key=lambda record: record.index)
    metrics = request_metrics(completed, duration)
    metrics.update(
        {
            "max_concurrency_observed": maximum_concurrency,
            "max_physical_pages_used": maximum_pages_used,
            "resident_gpu_bytes": resident_bytes,
            "peak_gpu_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "engine_profile_ms": engine._profiler.summary(),
            "final_block_state": engine.block_manager.stats(),
            "cuda": cuda_device_info(device) if device.type == "cuda" else None,
        }
    )
    if hasattr(engine.model_runner, "kv_cache"):
        cache = engine.model_runner.kv_cache
        metrics["physical_kv_cache"] = {
            "shape": list(cache.shape),
            "bytes": cache.numel() * cache.element_size(),
        }
    if save_detailed:
        metrics["requests"] = [asdict(record) for record in completed]
    return metrics
