class SequenceStatus:
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    PREEMPTED = "PREEMPTED"


class Sequence:
    """Per-request state: tokens, KV block mapping, and lifecycle status.

    Each sequence tracks its token IDs, a logical-to-physical block table
    for the KV cache, and counters for computed/cached/generated tokens.
    The scheduler advances it through WAITING -> RUNNING -> FINISHED.

    Attributes:
        seq_id: Monotonically increasing sequence identifier.
        request_id: User-assigned or auto-generated request label.
        status: Current lifecycle state (WAITING/RUNNING/FINISHED/PREEMPTED).
        token_ids: Full list of token IDs (prompt prefix + generated suffix).
        prompt_len: Number of tokens in the original prompt.
        num_computed_tokens: Tokens processed by prefill so far.
        num_cached_tokens: Tokens whose KV blocks were reused from prefix cache.
        num_generated_tokens: Tokens generated during decode.
        block_table: List of physical block IDs mapping logical token positions.
        enqueue_time, first_token_time, completion_time: Wall-clock timestamps.
    """

    counter: int = 0

    def __init__(self, token_ids, sampling_params, request_id=None):
        self.seq_id = Sequence.counter
        Sequence.counter += 1

        self.request_id = request_id or f"req-{self.seq_id}"
        self.status = SequenceStatus.WAITING
        self.sampling_params = sampling_params

        self.token_ids = list(token_ids)
        self.prompt_len = len(token_ids)
        self.num_computed_tokens = 0
        self.num_generated_tokens = 0
        self.num_cached_tokens = 0

        self.block_table = []

        self.enqueue_time = None
        self.first_token_time = None
        self.completion_time = None

    @property
    def last_token(self):
        return self.token_ids[-1]

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def uncomputed_tokens(self):
        return self.prompt_len - self.num_computed_tokens

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.num_generated_tokens += 1

    def __repr__(self):
        return (
            f"Sequence(id={self.seq_id}, req={self.request_id}, "
            f"status={self.status}, tokens={len(self.token_ids)}, "
            f"computed={self.num_computed_tokens}, "
            f"gen={self.num_generated_tokens})"
        )
