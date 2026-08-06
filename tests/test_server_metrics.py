from nanovllm_pp.observability.profiler import (
    EngineProfiler,
    RequestTracker,
    serving_metrics_snapshot,
)


class _Scheduler:
    @staticmethod
    def debug_state():
        return {
            "waiting": 2,
            "running": 3,
            "blocks": {"free_blocks": 7, "used_blocks": 5, "total_blocks": 12},
        }


class _Engine:
    scheduler = _Scheduler()
    step_counter = 9
    _profiler = EngineProfiler()
    _request_tracker = RequestTracker()


def test_metrics_snapshot_includes_serving_and_latency_metrics(monkeypatch):
    times = iter([0.0, 0.1, 0.5])
    monkeypatch.setattr("time.perf_counter", lambda: next(times))
    engine = _Engine()
    engine._request_tracker = RequestTracker()
    engine._request_tracker.record_enqueue("request-1")
    engine._request_tracker.record_first_token("request-1")
    engine._request_tracker.record_completion("request-1", num_tokens=5)

    metrics = serving_metrics_snapshot(engine)

    assert metrics["waiting_requests"] == 2
    assert metrics["used_logical_blocks"] == 5
    assert metrics["engine_steps"] == 9
    assert metrics["profiled_requests"] == 1
    assert metrics["request_latency_ms"]["ttft"]["p50"] == 100.0
    assert metrics["request_latency_ms"]["tpot"]["p50"] == 100.0
    assert metrics["request_latency_ms"]["e2e"]["p50"] == 500.0
