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
        assert self.temperature >= 0.0, "temperature must be >= 0"
        assert self.max_tokens > 0, "max_tokens must be > 0"
        assert 0.0 <= self.top_p <= 1.0, "top_p must be in [0, 1]"
        if isinstance(self.stop, str):
            self.stop = [self.stop]
