from dataclasses import dataclass
import torch

from .proposer import NGramProposer, GreedyVerifier


@dataclass
class SpecMetrics:
    proposals_made: int = 0
    proposals_empty: int = 0
    tokens_proposed: int = 0
    tokens_accepted: int = 0
    target_forwards: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.tokens_proposed == 0:
            return 0.0
        return self.tokens_accepted / self.tokens_proposed

    def report(self) -> str:
        return (
            f"accept_rate={self.acceptance_rate:.2f}, "
            f"accepted={self.tokens_accepted}/{self.tokens_proposed}, "
            f"target_fwds={self.target_forwards}, "
            f"empty_proposals={self.proposals_empty}/{self.proposals_made}"
        )


class SpecDecodeManager:
    """Orchestrates speculative decoding: propose, forward, verify.

    For each step:
      1. Proposer generates up to max_spec_tokens candidates.
      2. If no proposal, fall back to standard single-token decode.
      3. Target model runs one forward pass over proposal tokens.
      4. Verifier checks acceptance; accepted tokens are appended.

    Metrics track acceptance rate, target forwards saved, and proposal
    quality for tuning n-gram parameters.
    """

    def __init__(self, model_runner, max_spec_tokens=4):
        self.model_runner = model_runner
        self.max_spec_tokens = max_spec_tokens
        self.proposer = NGramProposer(min_ngram=2, max_ngram=5)
        self.verifier = GreedyVerifier()
        self.metrics = SpecMetrics()

    def step(self, seq) -> int:
        """Run one speculative step. Returns number of tokens generated."""
        proposal = self.proposer.propose(seq, self.max_spec_tokens)
        self.metrics.proposals_made += 1

        if not proposal.token_ids:
            self.metrics.proposals_empty += 1
            token_id = self.model_runner.run_decode(seq)
            self.metrics.target_forwards += 1
            return 1

        self.metrics.tokens_proposed += len(proposal.token_ids)

        ids = torch.tensor(
            proposal.token_ids, dtype=torch.long, device=self.model_runner.device
        ).unsqueeze(0)
        with torch.no_grad():
            output = self.model_runner.model(ids, use_cache=False)
            logits = output.logits[0]

        self.metrics.target_forwards += 1

        accepted, _ = self.verifier.verify(logits, proposal)
        self.metrics.tokens_accepted += len(accepted)

        for tid in accepted:
            seq.append_token(tid)

        return len(accepted)
