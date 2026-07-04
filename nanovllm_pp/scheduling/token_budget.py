from dataclasses import dataclass


@dataclass
class ScheduledItem:
    """A single unit of scheduler work: prefill-N-tokens or decode-1-token."""

    seq: object
    kind: str
    num_tokens: int


class TokenBudget:
    """Token budget for mixed prefill/decode scheduling.

    Limits the number of tokens processed per engine step rather than
    the number of sequences. Prevents long prompts from starving short
    ones during chunked prefill.
    """

    def __init__(
        self,
        max_tokens: int,
        max_seqs: int = 256,
        decode_priority: bool = True,
        max_prefill_chunk: int = 512,
    ):
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.decode_priority = decode_priority
        self.max_prefill_chunk = max_prefill_chunk
        self.remaining = max_tokens
        self.scheduled_count = 0

    def can_fit(self, num_tokens: int) -> bool:
        return num_tokens <= self.remaining and self.scheduled_count < self.max_seqs

    def consume(self, num_tokens: int):
        self.remaining -= num_tokens
        self.scheduled_count += 1

    def prefill_chunk_size(self, remaining_prompt: int) -> int:
        return min(remaining_prompt, self.remaining, self.max_prefill_chunk)

    def reset(self):
        self.remaining = self.max_tokens
        self.scheduled_count = 0
