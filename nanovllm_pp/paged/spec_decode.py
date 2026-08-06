import torch

from ..spec_decode.manager import SpecDecodeManager


class PagedSpecDecodeManager(SpecDecodeManager):
    """Exact n-gram speculation that commits only accepted physical pages."""

    def step(self, seq) -> list[int]:
        remaining = seq.sampling_params.max_tokens - seq.num_generated_tokens
        if remaining <= 0:
            return []
        if remaining == 1 or seq.sampling_params.stop:
            return self._fallback(seq)

        max_candidates = min(self.max_spec_tokens, remaining - 1)
        proposal = self.proposer.propose(seq, max_candidates)
        self.metrics.proposals_made += 1
        if not proposal.token_ids:
            self.metrics.proposals_empty += 1
            return self._fallback(seq)

        self.metrics.tokens_proposed += len(proposal.token_ids)
        query_tokens = [seq.last_token, *proposal.token_ids]
        logits = self.model_runner.verify_tokens(seq, query_tokens)
        self.metrics.target_forwards += 1
        accepted, correction = self.verifier.verify(logits[:-1], proposal)
        self.metrics.tokens_accepted += len(accepted)

        emitted = list(accepted)
        if correction is None:
            emitted.append(int(torch.argmax(logits[len(proposal.token_ids)]).item()))
        else:
            emitted.append(correction)

        terminal_ids = set(seq.sampling_params.stop_token_ids or [])
        if not seq.sampling_params.ignore_eos and seq.eos_token_id is not None:
            terminal_ids.add(seq.eos_token_id)
        terminal_index = next(
            (
                index
                for index, token_id in enumerate(emitted)
                if token_id in terminal_ids
            ),
            None,
        )
        if terminal_index is not None:
            emitted = emitted[: terminal_index + 1]

        start = len(seq.token_ids) - 1
        top_n = seq.sampling_params.logprobs
        for index, token_id in enumerate(emitted):
            seq.append_token(
                token_id,
                logprob=self._token_logprob(logits[index], token_id, top_n),
            )
            self.model_runner.block_manager.record_token(seq, token_id)

        # The target forward contains the previous last token plus proposals.
        # The cache invariant excludes the final emitted token, so exactly
        # len(emitted) query positions are valid regardless of accept/reject.
        self.model_runner.block_manager.mark_computed(seq, start, len(emitted))
        self.metrics.tokens_emitted += len(emitted)
        return emitted
