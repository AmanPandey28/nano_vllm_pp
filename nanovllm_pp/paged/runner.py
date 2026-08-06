import torch

from ..engine.model_runner import ModelRunner
from .context import PagedBatchMetadata


class PagedModelRunner(ModelRunner):
    """Batched runner backed by one global, physically paged K/V tensor."""

    def __init__(
        self,
        model,
        model_config,
        engine_config,
        device,
        block_manager,
        num_kv_blocks,
    ):
        super().__init__(model, model_config, engine_config, device, block_manager)
        cache_shape = (
            2,
            model_config.num_layers,
            num_kv_blocks,
            engine_config.block_size,
            model_config.num_kv_heads,
            model_config.head_dim,
        )
        parameter = next(model.parameters())
        self.kv_cache = torch.empty(cache_shape, device=device, dtype=parameter.dtype)
        self.kv_keys = self.kv_cache[0]
        self.kv_values = self.kv_cache[1]

    def _metadata(self, work, *, is_decode):
        """Build device tensors from ``(sequence, start, token_ids)`` work."""
        flat_tokens = []
        flat_positions = []
        flat_slots = []
        query_starts = [0]
        context_lens = []
        maximum_pages = 1

        for seq, start, token_ids in work:
            end = start + len(token_ids)
            flat_tokens.extend(token_ids)
            flat_positions.extend(range(start, end))
            flat_slots.extend(self.block_manager.slot_mapping(seq, range(start, end)))
            query_starts.append(len(flat_tokens))
            context_lens.append(end)
            pages = self.block_manager.required_blocks(end)
            maximum_pages = max(maximum_pages, pages)

        block_tables = torch.full(
            (len(work), maximum_pages),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        for row, (seq, _, _) in enumerate(work):
            page_ids = seq.block_table[:maximum_pages]
            block_tables[row, : len(page_ids)] = torch.tensor(
                page_ids, dtype=torch.int32, device=self.device
            )

        metadata = PagedBatchMetadata(
            positions=torch.tensor(
                flat_positions, dtype=torch.long, device=self.device
            ),
            slot_mapping=torch.tensor(flat_slots, dtype=torch.long, device=self.device),
            query_start_loc=torch.tensor(
                query_starts, dtype=torch.int32, device=self.device
            ),
            context_lens=torch.tensor(
                context_lens, dtype=torch.int32, device=self.device
            ),
            block_tables=block_tables,
            is_decode=is_decode,
            block_size=self.engine_config.block_size,
            query_starts=tuple(query_starts),
            context_lengths=tuple(context_lens),
            max_context_len=max(context_lens),
        )
        input_ids = torch.tensor(flat_tokens, dtype=torch.long, device=self.device)
        return input_ids, metadata

    def _forward(self, work, *, is_decode, logit_indices):
        input_ids, metadata = self._metadata(work, is_decode=is_decode)
        indices = torch.tensor(logit_indices, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            return self.model(
                input_ids,
                metadata,
                self.kv_cache,
                logit_indices=indices,
            )

    def run_prefill_batch(self, items):
        work = []
        logit_indices = []
        offset = 0
        for item in items:
            seq = item.seq
            start = seq.num_computed_tokens
            end = min(start + item.num_tokens, seq.prompt_len)
            token_ids = seq.token_ids[start:end]
            if not token_ids:
                raise RuntimeError("prefill scheduled with no uncomputed prompt tokens")
            work.append((seq, start, token_ids))
            offset += len(token_ids)
            logit_indices.append(offset - 1)

        logits = self._forward(work, is_decode=False, logit_indices=logit_indices)
        results = []
        for row, (item, (_, start, token_ids)) in enumerate(zip(items, work)):
            seq = item.seq
            self.block_manager.mark_computed(seq, start, len(token_ids))
            if start + len(token_ids) >= seq.prompt_len:
                token_id = self._sample_and_append(seq, logits[row])
                results.append((seq, token_id))
        return results

    def _run_decode_sequences(self, sequences):
        work = [(seq, len(seq.token_ids) - 1, [seq.last_token]) for seq in sequences]
        logits = self._forward(
            work,
            is_decode=True,
            logit_indices=list(range(len(sequences))),
        )
        results = []
        for row, (seq, start, _) in enumerate(work):
            self.block_manager.mark_computed(seq, start, 1)
            results.append(self._sample_and_append(seq, logits[row]))
        return results

    def run_decode(self, seq):
        return self._run_decode_sequences([seq])[0]

    def verify_tokens(self, seq, token_ids):
        """Run one target forward for speculative verification.

        The caller decides which speculative K/V writes are logically valid and
        marks only that accepted prefix as computed.
        """
        start = len(seq.token_ids) - 1
        work = [(seq, start, list(token_ids))]
        return self._forward(
            work,
            is_decode=False,
            logit_indices=list(range(len(token_ids))),
        )

    def run(self, scheduled_batch):
        results = []
        prefill_items = scheduled_batch.prefill_items()
        if prefill_items:
            results.extend(self.run_prefill_batch(prefill_items))

        regular_decode = []
        for item in scheduled_batch.decode_items():
            seq = item.seq
            if (
                self.spec_decode_manager is not None
                and seq.sampling_params.temperature == 0
            ):
                emitted = self.spec_decode_manager.step(seq)
                results.append((seq, emitted[-1]))
            else:
                regular_decode.append(seq)
        if regular_decode:
            tokens = self._run_decode_sequences(regular_decode)
            results.extend(zip(regular_decode, tokens))
        return results
