from dataclasses import dataclass
import torch


@dataclass
class Proposal:
    token_ids: list[int]


class Proposer:
    """Interface for generating token proposals during speculative decoding."""

    def propose(self, seq, max_tokens: int) -> Proposal:
        raise NotImplementedError


class NGramProposer(Proposer):
    """Finds matching n-gram suffixes in previously generated tokens.

    Searches backward through the sequence for a suffix match of length n
    (from max_ngram down to min_ngram). If found, proposes the tokens
    that followed that occurrence.

    This requires no second model — pure pattern matching on the existing
    token stream — making it the simplest form of speculative decoding.
    """

    def __init__(self, min_ngram=2, max_ngram=5):
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram

    def propose(self, seq, max_tokens: int) -> Proposal:
        tokens = seq.token_ids
        for n in range(min(self.max_ngram, len(tokens) - 1), self.min_ngram - 1, -1):
            suffix = tokens[-n:]
            for i in range(len(tokens) - n - 1, -1, -1):
                if tokens[i : i + n] == suffix:
                    start = i + n
                    proposal = tokens[start : start + max_tokens]
                    if proposal:
                        return Proposal(list(proposal))
        return Proposal([])


class GreedyVerifier:
    """Verifies proposed tokens against target model logits via argmax match.

    For each proposed token position, the target model's greedy prediction
    is compared against the proposal. Matching tokens are accepted; the
    first mismatch triggers rejection and the sequence stops there.

    Returns (accepted_tokens, rejection_token).
    """

    def verify(
        self, target_logits: torch.Tensor, proposal: Proposal
    ) -> tuple[list[int], int | None]:
        accepted = []
        rejection_token = None
        for i, proposed in enumerate(proposal.token_ids):
            target_token = int(torch.argmax(target_logits[i]))
            if proposed == target_token:
                accepted.append(proposed)
            else:
                accepted.append(target_token)
                rejection_token = target_token
                break
        return accepted, rejection_token
