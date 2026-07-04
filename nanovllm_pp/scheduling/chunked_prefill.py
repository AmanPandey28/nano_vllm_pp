from .token_budget import TokenBudget, ScheduledItem


class MixedBatchScheduler:
    """Schedules mixed prefill/decode batches under a token budget.

    Algorithm:
      1. Phase A (decode): iterate running queue, schedule 1 token each.
      2. Phase B (prefill): iterate waiting queue, schedule chunks.
         Sequences stay in waiting until their full prompt is processed.
         Block allocation happens on first admission.

    With disable_chunked_prefill, prefill must complete for all admitted
    sequences before any decode can run.
    """

    def __init__(self, config, block_manager):
        self.config = config
        self.block_manager = block_manager
        self.budget_cls = TokenBudget

    def build_items(self, waiting, running) -> list[ScheduledItem]:
        budget = self.budget_cls(
            max_tokens=self.config.max_num_batched_tokens,
            max_seqs=self.config.max_num_seqs,
            decode_priority=self.config.decode_priority,
            max_prefill_chunk=self.config.max_prefill_chunk_size,
        )
        items = []

        if budget.decode_priority:
            self._schedule_decode(running, budget, items)

        self._schedule_prefill_chunks(waiting, budget, items)

        if not budget.decode_priority:
            self._schedule_decode(running, budget, items)

        return items

    def _schedule_decode(self, running, budget, items):
        for seq in list(running):
            if not budget.can_fit(1):
                break
            if not self.block_manager.can_append(seq):
                continue
            self.block_manager.may_append(seq)
            items.append(ScheduledItem(seq=seq, kind="decode", num_tokens=1))
            budget.consume(1)

    def _schedule_prefill_chunks(self, waiting, budget, items):
        while waiting and budget.remaining > 0:
            seq = waiting[0]
            remaining = seq.uncomputed_tokens

            if remaining <= 0:
                waiting.popleft()
                continue

            if self.config.enable_chunked_prefill:
                if seq.num_computed_tokens == 0:
                    if not self.block_manager.can_allocate(seq):
                        break
                    self.block_manager.allocate(seq)

                chunk = budget.prefill_chunk_size(remaining)
                if chunk <= 0:
                    break
                if not budget.can_fit(chunk):
                    break

                items.append(ScheduledItem(seq=seq, kind="prefill", num_tokens=chunk))
                budget.consume(chunk)

                if chunk >= remaining:
                    waiting.popleft()
                else:
                    break
            else:
                if not budget.can_fit(remaining):
                    break
                if not self.block_manager.can_allocate(seq):
                    break
                waiting.popleft()
                self.block_manager.allocate(seq)
                items.append(
                    ScheduledItem(seq=seq, kind="prefill", num_tokens=remaining)
                )
                budget.consume(remaining)
