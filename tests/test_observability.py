import json

from nanovllm_pp.observability.profiler import EngineProfiler, RequestTracker
from nanovllm_pp.observability.trace import EngineTrace


def test_profiler_keeps_mixed_work_separate():
    profiler = EngineProfiler()
    profiler.start_step(1)
    profiler.tick("schedule")
    profiler.tick("mixed")
    profiler.tick("postprocess")
    profiler.end_step(num_prefill=1, num_decode=1)

    summary = profiler.summary()
    assert summary["steps"] == 1
    assert "avg_mixed_ms" in summary
    assert profiler.history[0].num_prefill == 1
    assert profiler.history[0].num_decode == 1


def test_request_tracker_tpot_uses_number_of_inter_token_intervals(monkeypatch):
    times = iter([0.0, 0.1, 0.5])
    monkeypatch.setattr("time.perf_counter", lambda: next(times))
    tracker = RequestTracker()
    tracker.record_enqueue("r")
    tracker.record_first_token("r")
    tracker.record_completion("r", num_tokens=5)

    report = tracker.report()
    assert report["ttft"]["p50"] == 100.0
    assert report["tpot"]["p50"] == 100.0
    assert report["e2e"]["p50"] == 500.0


def test_trace_writes_json_lines(tmp_path):
    trace = EngineTrace()
    trace.add(step=1, event_type="schedule", mode="prefill", seq_ids=[3])
    path = tmp_path / "trace.jsonl"
    trace.write_jsonl(path)

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["event_type"] == "schedule"
    assert event["seq_ids"] == [3]
