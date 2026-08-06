from .token_budget import ScheduledItem, TokenBudget


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
            remaining = max(
                1, seq.sampling_params.max_tokens - seq.num_generated_tokens
            )
            lookahead = 1
            if (
                self.config.enable_spec_decode
                and seq.sampling_params.temperature == 0
                and not seq.sampling_params.stop
            ):
                lookahead = min(1 + self.config.max_spec_tokens, remaining)
            if not budget.can_fit(lookahead):
                continue
            if not self.block_manager.can_append(seq, lookahead):
                continue
            self.block_manager.may_append(seq, lookahead)
            items.append(ScheduledItem(seq=seq, kind="decode", num_tokens=lookahead))
            budget.consume(lookahead)

    def _schedule_prefill_chunks(self, waiting, budget, items):
        # Consider every waiting sequence at most once per engine iteration.
        # A partially-prefilled long request moves to the back, allowing short
        # prompts to consume the remaining budget instead of being head-of-line
        # blocked forever by the first deque entry.
        for _ in range(len(waiting)):
            if not waiting or budget.remaining <= 0:
                break
            seq = waiting[0]
            if seq.block_table:
                remaining = seq.uncomputed_tokens
            else:
                remaining = seq.prompt_len - self.block_manager.cached_prefix_tokens(
                    seq
                )

            if remaining <= 0:
                waiting.popleft()
                continue

            if self.config.enable_chunked_prefill:
                if not seq.block_table:
                    if not self.block_manager.can_allocate(seq, num_new_tokens=1):
                        waiting.rotate(-1)
                        continue
                    self.block_manager.allocate(seq, num_new_tokens=1)
                    seq.num_computed_tokens = seq.num_cached_tokens
                    remaining = seq.uncomputed_tokens

                chunk = budget.prefill_chunk_size(remaining)
                if chunk <= 0:
                    break
                if not budget.can_fit(chunk):
                    waiting.rotate(-1)
                    continue

                items.append(ScheduledItem(seq=seq, kind="prefill", num_tokens=chunk))
                budget.consume(chunk)

                if chunk >= remaining:
                    waiting.popleft()
                else:
                    waiting.rotate(-1)
            else:
                if not budget.can_fit(remaining):
                    waiting.rotate(-1)
                    continue
                if not self.block_manager.can_allocate(seq, num_new_tokens=1):
                    waiting.rotate(-1)
                    continue
                waiting.popleft()
                self.block_manager.allocate(seq, num_new_tokens=1)
                seq.num_computed_tokens = seq.num_cached_tokens
                remaining = seq.uncomputed_tokens
                items.append(
                    ScheduledItem(seq=seq, kind="prefill", num_tokens=remaining)
                )
                budget.consume(remaining)
