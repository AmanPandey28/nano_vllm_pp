from collections import deque

from ..engine.sequence import SequenceStatus
from ..scheduling.chunked_prefill import MixedBatchScheduler
from ..scheduling.token_budget import ScheduledItem


class ScheduledBatch:
    """Container for work items produced by the scheduler.

    Each item is a ScheduledItem with a sequence reference, a kind
    ('prefill' or 'decode'), and a token count.
    """

    def __init__(self, items):
        self.items = items

    @property
    def seqs(self):
        return [item.seq for item in self.items]

    @property
    def is_prefill(self):
        return all(i.kind == "prefill" for i in self.items)

    @property
    def is_decode(self):
        return all(i.kind == "decode" for i in self.items)

    def prefill_items(self):
        return [i for i in self.items if i.kind == "prefill"]

    def decode_items(self):
        return [i for i in self.items if i.kind == "decode"]

    def __bool__(self):
        return len(self.items) > 0

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        p = sum(1 for i in self.items if i.kind == "prefill")
        d = sum(1 for i in self.items if i.kind == "decode")
        return f"ScheduledBatch(prefill={p}, decode={d})"


class Scheduler:
    """Continuous batching scheduler with optional chunked prefill.

    Maintains two FIFO queues:
      - waiting: sequences that have not yet been prefilled.
      - running: sequences actively generating tokens.

    When enable_chunked_prefill is False:
      Prefill completes fully before any decode runs.

    When enable_chunked_prefill is True:
      Decode gets priority on the token budget. Remaining budget is used
      for prefill chunks, allowing decode to interleave with long prompts.
    """

    def __init__(self, config, block_manager):
        self.config = config
        self.block_manager = block_manager
        self.waiting = deque()
        self.running = deque()
        self.mixed_scheduler = MixedBatchScheduler(config, block_manager)

    def add(self, seq):
        self.waiting.append(seq)

    def has_unfinished(self):
        return bool(self.waiting) or bool(self.running)

    def schedule(self):
        if self.config.enable_chunked_prefill:
            items = self.mixed_scheduler.build_items(self.waiting, self.running)
            return ScheduledBatch(items)

        if self.waiting:
            batch = self._schedule_prefill_only()
            if batch:
                return batch
            # A waiting prompt may need pages owned by a running request. Let
            # decode finish that request instead of reporting false deadlock.
            return self._schedule_decode_only()
        if self.running:
            return self._schedule_decode_only()
        return ScheduledBatch([])

    def _schedule_prefill_only(self):
        scheduled = []
        budget = self.config.max_num_batched_tokens
        max_seqs = self.config.max_num_seqs

        for _ in range(len(self.waiting)):
            if not self.waiting or len(scheduled) >= max_seqs:
                break
            seq = self.waiting[0]
            cached = self.block_manager.cached_prefix_tokens(seq)
            remaining = seq.prompt_len - cached
            if remaining > budget:
                self.waiting.rotate(-1)
                continue
            # The final prefill forward emits the first generated token, so its
            # logical cache slot must be admitted together with the prompt.
            if not self.block_manager.can_allocate(seq, num_new_tokens=1):
                self.waiting.rotate(-1)
                continue
            self.block_manager.allocate(seq, num_new_tokens=1)
            self.waiting.popleft()
            seq.num_computed_tokens = seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            scheduled.append(
                ScheduledItem(seq=seq, kind="prefill", num_tokens=remaining)
            )
            budget -= remaining

        return ScheduledBatch(scheduled)

    def _schedule_decode_only(self):
        scheduled = []
        token_budget = self.config.max_num_batched_tokens
        for seq in list(self.running):
            if len(scheduled) >= self.config.max_num_seqs:
                break
            lookahead = self._decode_lookahead(seq)
            if lookahead > token_budget:
                continue
            if not self.block_manager.can_append(seq, lookahead):
                continue
            self.block_manager.may_append(seq, lookahead)
            scheduled.append(
                ScheduledItem(seq=seq, kind="decode", num_tokens=lookahead)
            )
            token_budget -= lookahead
        return ScheduledBatch(scheduled)

    def _decode_lookahead(self, seq):
        remaining = max(1, seq.sampling_params.max_tokens - seq.num_generated_tokens)
        if (
            self.config.enable_spec_decode
            and seq.sampling_params.temperature == 0
            and not seq.sampling_params.stop
        ):
            return min(1 + self.config.max_spec_tokens, remaining)
        return 1

    @staticmethod
    def _finish_reason(seq):
        if seq.stop_matched is not None:
            return "stop"
        if seq.last_token in (seq.sampling_params.stop_token_ids or []):
            return "stop"
        if (
            not seq.sampling_params.ignore_eos
            and seq.eos_token_id is not None
            and seq.last_token == seq.eos_token_id
        ):
            return "stop"
        if seq.num_generated_tokens >= seq.sampling_params.max_tokens:
            return "length"
        return None

    def _finish(self, seq, reason):
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        self.block_manager.deallocate(seq)
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)
        if hasattr(seq, "_kv_cache"):
            del seq._kv_cache

    def abort(self, seq, reason="cancelled"):
        """Release waiting, partially prefilled, or running request resources."""
        if seq.status != SequenceStatus.FINISHED:
            self._finish(seq, reason)

    def postprocess(self, scheduled_batch, output_tokens):
        """Update sequence state after model execution.

        For prefill items: increment computed tokens; if prompt is complete,
        move the sequence to the running queue.

        For decode items: check EOS and max_tokens limits; finish and
        deallocate blocks when a sequence is complete.
        """
        finished = []
        for item in scheduled_batch.items:
            seq = item.seq

            if item.kind == "prefill":
                seq.num_computed_tokens += item.num_tokens
                if seq.num_computed_tokens >= seq.prompt_len:
                    reason = self._finish_reason(seq)
                    if reason is not None:
                        self._finish(seq, reason)
                        finished.append(seq)
                    else:
                        if seq.status == SequenceStatus.WAITING:
                            seq.status = SequenceStatus.RUNNING
                        self.running.append(seq)

            elif item.kind == "decode":
                reason = self._finish_reason(seq)
                if reason is not None:
                    self._finish(seq, reason)
                    finished.append(seq)

        return finished

    def debug_state(self):
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "blocks": self.block_manager.stats(),
        }
