from dataclasses import dataclass

import torch

from .proposer import GreedyVerifier, NGramProposer


@dataclass
class SpecMetrics:
    proposals_made: int = 0
    proposals_empty: int = 0
    tokens_proposed: int = 0
    tokens_accepted: int = 0
    target_forwards: int = 0
    tokens_emitted: int = 0

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
            f"emitted={self.tokens_emitted}, "
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

    @staticmethod
    def _cache_length(cache) -> int:
        if cache is None:
            return 0
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
        if isinstance(cache, (tuple, list)) and cache:
            return int(cache[0][0].shape[-2])
        raise TypeError(f"unsupported KV cache type: {type(cache).__name__}")

    @staticmethod
    def _crop_cache(cache, length: int):
        if hasattr(cache, "crop"):
            cache.crop(length)
            return cache
        if isinstance(cache, (tuple, list)):
            cropped_layers = []
            for layer in cache:
                cropped_states = []
                for state in layer:
                    if isinstance(state, torch.Tensor) and state.ndim >= 3:
                        cropped_states.append(state[..., :length, :])
                    else:
                        cropped_states.append(state)
                cropped_layers.append(tuple(cropped_states))
            return tuple(cropped_layers)
        raise TypeError(f"unsupported KV cache type: {type(cache).__name__}")

    @staticmethod
    def _token_logprob(logits, token_id: int, top_n: int | None = None):
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        info = {"logprob": float(logprobs[token_id].item())}
        if top_n:
            values, indices = torch.topk(logprobs, min(top_n, logprobs.numel()))
            info["top_logprobs"] = {
                int(idx): float(value)
                for idx, value in zip(indices.tolist(), values.tolist())
            }
        return info

    def _fallback(self, seq) -> list[int]:
        token_id = self.model_runner.run_decode(seq)
        self.metrics.target_forwards += 1
        self.metrics.tokens_emitted += 1
        return [token_id]

    def step(self, seq) -> list[int]:
        """Run one exact greedy speculative step and return emitted token IDs.

        The target cache invariant matches normal decoding: the cache contains
        every sequence token except the final emitted token.  On rejection, the
        over-computed suffix is cropped before the target correction is appended.
        """
        remaining = seq.sampling_params.max_tokens - seq.num_generated_tokens
        if remaining <= 0:
            return []
        # Stop strings require decoding text after every token.  Falling back
        # to one token per step prevents a multi-token speculative burst from
        # stepping past a string boundary.
        if (
            remaining == 1
            or getattr(seq, "_kv_cache", None) is None
            or seq.sampling_params.stop
        ):
            return self._fallback(seq)

        max_candidates = min(self.max_spec_tokens, remaining - 1)
        proposal = self.proposer.propose(seq, max_candidates)
        self.metrics.proposals_made += 1

        if not proposal.token_ids:
            self.metrics.proposals_empty += 1
            return self._fallback(seq)

        self.metrics.tokens_proposed += len(proposal.token_ids)
        old_cache = seq._kv_cache
        old_cache_len = self._cache_length(old_cache)
        # The cache excludes seq.last_token.  Feeding it before the proposals
        # aligns logits[i] with proposal[i].
        ids = torch.tensor(
            [seq.last_token, *proposal.token_ids],
            dtype=torch.long,
            device=self.model_runner.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            output = self.model_runner.model(
                ids, past_key_values=old_cache, use_cache=True
            )
            logits = output.logits[0]

        self.metrics.target_forwards += 1
        accepted, correction = self.verifier.verify(logits[:-1], proposal)
        self.metrics.tokens_accepted += len(accepted)

        emitted = list(accepted)
        if correction is not None:
            emitted.append(correction)
            keep_length = old_cache_len + 1 + len(accepted)
            seq._kv_cache = self._crop_cache(output.past_key_values, keep_length)
        else:
            # All proposals matched.  The final row predicts one bonus token;
            # the output cache already contains last_token + every proposal.
            bonus = int(torch.argmax(logits[len(proposal.token_ids)]).item())
            emitted.append(bonus)
            seq._kv_cache = output.past_key_values

        # A speculative pass can predict beyond a token that terminates the
        # request.  Keep only the prefix through that token and crop the cache
        # so it once again excludes the final emitted token.
        terminal_ids = set(seq.sampling_params.stop_token_ids or [])
        if not seq.sampling_params.ignore_eos and seq.eos_token_id is not None:
            terminal_ids.add(seq.eos_token_id)
        terminal_index = next(
            (i for i, token_id in enumerate(emitted) if token_id in terminal_ids),
            None,
        )
        if terminal_index is not None:
            emitted = emitted[: terminal_index + 1]
            seq._kv_cache = self._crop_cache(
                output.past_key_values, old_cache_len + len(emitted)
            )

        top_n = seq.sampling_params.logprobs
        for i, token_id in enumerate(emitted):
            seq.append_token(
                token_id,
                logprob=self._token_logprob(logits[i], token_id, top_n),
            )
            block_manager = self.model_runner.block_manager
            if block_manager is not None and seq.block_table:
                block_manager.record_token(seq, token_id)
        self.metrics.tokens_emitted += len(emitted)
        return emitted
