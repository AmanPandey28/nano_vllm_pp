from dataclasses import dataclass, asdict
import json
import time


@dataclass
class TraceEvent:
    step: int
    event_type: str
    timestamp: float
    mode: str | None = None
    seq_ids: list[int] | None = None
    num_tokens: int | None = None
    free_blocks: int | None = None
    used_blocks: int | None = None
    latency_ms: float | None = None
    extra: dict | None = None


class EngineTrace:
    """Structured event logging for offline analysis of engine behavior.

    Records scheduler decisions, KV block allocations, and per-step
    metrics as JSONL for post-hoc visualization and debugging.
    """

    def __init__(self):
        self.events: list[TraceEvent] = []
        self._start_time = time.time()

    def add(self, **kwargs):
        kwargs.setdefault("timestamp", time.time() - self._start_time)
        self.events.append(TraceEvent(**kwargs))

    def write_jsonl(self, path: str):
        with open(path, "w") as f:
            for e in self.events:
                f.write(json.dumps(asdict(e), default=str) + "\n")

    def summary(self) -> dict:
        if not self.events:
            return {}
        types = {}
        for e in self.events:
            types[e.event_type] = types.get(e.event_type, 0) + 1
        return {
            "total_events": len(self.events),
            "duration_s": (self.events[-1].timestamp - self.events[0].timestamp)
            if len(self.events) > 1
            else 0,
            "event_types": types,
        }
