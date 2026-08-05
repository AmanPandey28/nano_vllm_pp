from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    seed: int | None = None
    logprobs: int | None = None
    prompt_logprobs: bool = False

    def __post_init__(self):
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.logprobs is not None and self.logprobs <= 0:
            raise ValueError("logprobs must be a positive integer or None")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0")
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.stop is not None and any(not s for s in self.stop):
            raise ValueError("stop strings must be non-empty")
        if self.stop_token_ids is not None:
            self.stop_token_ids = [int(token_id) for token_id in self.stop_token_ids]
