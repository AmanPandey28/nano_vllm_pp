import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class StepMetrics:
    step: int
    schedule_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    num_prefill: int = 0
    num_decode: int = 0
    num_prefill_tokens: int = 0
    free_blocks: int = 0
    waiting_queue: int = 0
    running_queue: int = 0


class EngineProfiler:
    """Per-step phase timing for engine bottleneck diagnosis.

    Instruments each engine step to measure wall-clock time spent in:
      - schedule: scheduler decision logic
      - prefill: model forward for prompt tokens
      - decode: model forward for single-token generation
      - postprocess: sequence state updates and block deallocation
    """

    def __init__(self):
        self.history: list[StepMetrics] = []
        self._current: StepMetrics | None = None
        self._t0 = 0.0

    def start_step(self, step: int):
        self._current = StepMetrics(step=step)
        self._t0 = time.time()

    def tick(self, phase: str):
        if self._current is None:
            return
        elapsed = (time.time() - self._t0) * 1000
        if phase == "schedule":
            self._current.schedule_ms = elapsed
        elif phase == "prefill":
            self._current.prefill_ms = elapsed
        elif phase == "decode":
            self._current.decode_ms = elapsed
        elif phase == "postprocess":
            self._current.postprocess_ms = elapsed
        self._t0 = time.time()

    def end_step(
        self,
        num_prefill=0,
        num_decode=0,
        num_prefill_tokens=0,
        free_blocks=0,
        waiting=0,
        running=0,
    ):
        if self._current is None:
            return
        self._current.total_ms = sum(
            [
                self._current.schedule_ms,
                self._current.prefill_ms,
                self._current.decode_ms,
                self._current.postprocess_ms,
            ]
        )
        self._current.num_prefill = num_prefill
        self._current.num_decode = num_decode
        self._current.num_prefill_tokens = num_prefill_tokens
        self._current.free_blocks = free_blocks
        self._current.waiting_queue = waiting
        self._current.running_queue = running
        self.history.append(self._current)
        self._current = None

    def summary(self) -> dict:
        if not self.history:
            return {}
        from statistics import mean

        return {
            "steps": len(self.history),
            "avg_schedule_ms": round(mean(h.schedule_ms for h in self.history), 3),
            "avg_prefill_ms": round(mean(h.prefill_ms for h in self.history), 3),
            "avg_decode_ms": round(mean(h.decode_ms for h in self.history), 3),
            "avg_postprocess_ms": round(
                mean(h.postprocess_ms for h in self.history), 3
            ),
            "avg_total_ms": round(mean(h.total_ms for h in self.history), 3),
            "bottleneck": self._bottleneck(),
        }

    def _bottleneck(self) -> str:
        parts = {
            "schedule": sum(h.schedule_ms for h in self.history),
            "prefill": sum(h.prefill_ms for h in self.history),
            "decode": sum(h.decode_ms for h in self.history),
            "postprocess": sum(h.postprocess_ms for h in self.history),
        }
        return max(parts, key=parts.get)

    def timeline(self) -> list[dict]:
        return [
            {
                "step": h.step,
                "schedule": round(h.schedule_ms, 3),
                "prefill": round(h.prefill_ms, 3),
                "decode": round(h.decode_ms, 3),
                "post": round(h.postprocess_ms, 3),
                "total": round(h.total_ms, 3),
                "waiting_q": h.waiting_queue,
                "running_q": h.running_queue,
            }
            for h in self.history
        ]

    def print_report(self):
        s = self.summary()
        print(f"\n{'=' * 50}")
        print(f"  Engine Phase Profile ({s['steps']} steps)")
        print(f"{'=' * 50}")
        print(f"  {'Phase':<15} {'Avg (ms)':<12} {'Pct':<8}")
        print(f"  {'-' * 35}")
        total = max(s["avg_total_ms"], 0.001)
        for phase, attr in [
            ("Schedule", "avg_schedule_ms"),
            ("Prefill", "avg_prefill_ms"),
            ("Decode", "avg_decode_ms"),
            ("Postprocess", "avg_postprocess_ms"),
        ]:
            val = s[attr]
            pct = val / total * 100
            bar = "█" * int(pct / 5)
            print(f"  {phase:<15} {val:<12.3f} {pct:5.1f}% {bar}")
        print(f"  {'-' * 35}")
        print(f"  Bottleneck: {s['bottleneck']}")
        print(f"{'=' * 50}\n")


class RequestTracker:
    """Per-request latency tracking: TTFT, TPOT, and E2E percentiles.

    Tracks each request from enqueue through first token to completion
    for measuring serving-level latency metrics.
    """

    def __init__(self):
        self.requests: dict[str, dict] = {}

    def record_enqueue(self, request_id: str):
        self.requests[request_id] = {
            "enqueue_t": time.time(),
            "first_token_t": None,
            "completion_t": None,
            "tokens_generated": 0,
        }

    def record_first_token(self, request_id: str):
        if (
            request_id in self.requests
            and self.requests[request_id]["first_token_t"] is None
        ):
            self.requests[request_id]["first_token_t"] = time.time()

    def record_completion(self, request_id: str, num_tokens: int = 0):
        if request_id in self.requests:
            r = self.requests[request_id]
            r["completion_t"] = time.time()
            if num_tokens > 0:
                r["tokens_generated"] = num_tokens

    def report(self) -> dict:
        ttft_vals, tpot_vals, e2e_vals = [], [], []
        for rid, r in self.requests.items():
            if r["first_token_t"]:
                ttft_vals.append((r["first_token_t"] - r["enqueue_t"]) * 1000)
            if r["completion_t"] and r["first_token_t"]:
                gen_time = (r["completion_t"] - r["first_token_t"]) * 1000
                tokens = max(r["tokens_generated"], 1)
                tpot_vals.append(gen_time / tokens)
            if r["completion_t"]:
                e2e_vals.append((r["completion_t"] - r["enqueue_t"]) * 1000)

        def stats(vals):
            if not vals:
                return {}
            vals = sorted(vals)
            n = len(vals)
            return {
                "p50": round(vals[n // 2], 1),
                "p95": round(vals[int(n * 0.95)], 1),
                "p99": round(vals[int(n * 0.99)], 1),
                "mean": round(sum(vals) / n, 1),
            }

        return {
            "num_requests": len(self.requests),
            "ttft": stats(ttft_vals),
            "tpot": stats(tpot_vals),
            "e2e": stats(e2e_vals),
        }

    def print_report(self):
        r = self.report()
        print(f"\n{'=' * 50}")
        print(f"  Request Latency Report ({r['num_requests']} requests)")
        print(f"{'=' * 50}")
        print(f"  {'Metric':<10} {'p50':<10} {'p95':<10} {'p99':<10} {'mean':<10}")
        print(f"  {'-' * 50}")
        for m in ["ttft", "tpot", "e2e"]:
            d = r[m]
            print(
                f"  {m.upper():<10} {d.get('p50', '—'):<10} "
                f"{d.get('p95', '—'):<10} {d.get('p99', '—'):<10} "
                f"{d.get('mean', '—'):<10}"
            )
        print(f"{'=' * 50}\n")


def diagnose_slowdown(profiler: EngineProfiler, tracker: RequestTracker) -> list[str]:
    """Automatic bottleneck detection combining engine and request metrics.

    Examines per-step phase timing and per-request latency percentiles
    to identify the most likely source of slowdown.
    """
    psum = profiler.summary()
    tsum = tracker.report()
    issues = []

    sched_pct = psum.get("avg_schedule_ms", 0) / max(psum.get("avg_total_ms", 1), 0.001)
    if sched_pct > 0.50:
        issues.append(
            "SCHEDULER: >50% of step time in scheduler. "
            "Check block allocation, queue iteration, or policy logic."
        )

    if psum.get("avg_prefill_ms", 0) > 100:
        issues.append(
            "PREFILL: >100ms average. Prompt too long or "
            "consider enabling chunked prefill."
        )

    ttft_p50 = tsum.get("ttft", {}).get("p50", 0)
    if ttft_p50 > 500:
        issues.append(
            f"TTFT: p50={ttft_p50}ms. Requests waiting too long "
            "in queue — check block availability and admission rate."
        )

    tpot_p50 = tsum.get("tpot", {}).get("p50", 0)
    if tpot_p50 > 50:
        issues.append(
            f"TPOT: p50={tpot_p50}ms per token. Decode is slow — "
            "consider quantization or speculative decoding."
        )

    if not issues:
        issues.append(
            "No obvious engine-level bottleneck. Use GPU profiling "
            "(PyTorch Profiler / Nsight Systems) for kernel analysis."
        )

    return issues
