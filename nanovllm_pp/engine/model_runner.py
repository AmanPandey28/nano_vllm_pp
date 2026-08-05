import torch

from ..config import ModelConfig


class Sampler:
    """Sampling layer — temperature scaling, top-k/top-p filtering, token selection."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def apply_temperature(self, logits, temperature):
        if temperature == 0:
            return logits
        return logits / temperature

    def apply_top_k(self, logits, top_k):
        if top_k <= 0:
            return logits
        top_k = min(top_k, logits.size(-1))
        threshold = logits.topk(top_k, dim=-1).values[..., -1, None]
        logits[logits < threshold] = float("-inf")
        return logits

    def apply_top_p(self, logits, top_p):
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        indices_to_remove = remove_mask.scatter(-1, sorted_indices, remove_mask)
        logits[indices_to_remove] = float("-inf")
        return logits

    def apply_min_p(self, logits, min_p):
        if min_p <= 0.0:
            return logits
        probs = torch.softmax(logits, dim=-1)
        threshold = probs.max(dim=-1, keepdim=True).values * min_p
        logits[probs < threshold] = float("-inf")
        return logits

    def __call__(self, logits, params, generator=None):
        # Filtering is in-place, so never mutate the model-owned output view.
        raw_logits = logits.float()
        sampling_logits = raw_logits.clone()
        sampling_logits = self.apply_temperature(sampling_logits, params.temperature)
        if params.top_k:
            sampling_logits = self.apply_top_k(sampling_logits, params.top_k)
        if params.top_p < 1.0:
            sampling_logits = self.apply_top_p(sampling_logits, params.top_p)
        if params.min_p > 0.0:
            sampling_logits = self.apply_min_p(sampling_logits, params.min_p)
        if params.temperature == 0:
            token_id = int(torch.argmax(sampling_logits, dim=-1).item())
        else:
            probs = torch.softmax(sampling_logits, dim=-1)
            token_id = int(
                torch.multinomial(probs, num_samples=1, generator=generator).item()
            )

        raw_logprobs = torch.log_softmax(raw_logits, dim=-1)
        info = {"logprob": float(raw_logprobs[token_id].item())}
        if params.logprobs:
            count = min(params.logprobs, raw_logprobs.numel())
            values, indices = torch.topk(raw_logprobs, count)
            info["top_logprobs"] = {
                int(idx): float(value)
                for idx, value in zip(indices.tolist(), values.tolist())
            }
        return token_id, info


class ModelRunner:
    """Loads the model and executes prefill/decode passes.

    Uses HuggingFace's built-in past_key_values for KV caching.
    Prefill processes prompt tokens in one or more chunks and stores the
    cache. Decode feeds a single token with the stored cache.

    Attributes:
        model: The loaded HuggingFace model (eval mode, on device).
        sampler: Token sampling logic.
        kv_keys/kv_values: Reserved for future paged KV cache integration.
    """

    def __init__(
        self,
        model,
        model_config: ModelConfig,
        engine_config,
        device="cuda",
        block_manager=None,
    ):
        self.model = model
        self.model_config = model_config
        self.engine_config = engine_config
        self.device = device
        self.block_manager = block_manager

        self.kv_keys = None
        self.kv_values = None
        self.sampler = Sampler(model_config.vocab_size)
        self.cuda_graphs = {}
        self.spec_decode_manager = None

    def _get_generator(self, seq):
        if seq.sampling_params.seed is None:
            return None
        generator = getattr(seq, "_sampling_generator", None)
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seq.sampling_params.seed)
            seq._sampling_generator = generator
        return generator

    def _sample_and_append(self, seq, logits):
        token_id, info = self.sampler(
            logits, seq.sampling_params, generator=self._get_generator(seq)
        )
        seq.append_token(
            token_id, logprob=info if seq.sampling_params.logprobs else None
        )
        if self.block_manager is not None and seq.block_table:
            self.block_manager.record_token(seq, token_id)
        return token_id

    def run_prefill(self, seq, num_tokens):
        """Process `num_tokens` tokens of the prompt.

        If this completes the full prompt, sample and append the first
        generated token. Stores past_key_values on the sequence for
        subsequent decode steps.
        """
        start = seq.num_computed_tokens
        end = min(start + num_tokens, seq.prompt_len)
        token_slice = seq.token_ids[start:end]
        if not token_slice:
            raise RuntimeError("prefill scheduled with no uncomputed prompt tokens")
        input_ids = torch.tensor(
            token_slice, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        past = getattr(seq, "_kv_cache", None)
        with torch.inference_mode():
            if past is None:
                output = self.model(input_ids, use_cache=True)
            else:
                # Carry cache state across chunks.  Omitting this argument would
                # make every chunk a new sequence and silently corrupt outputs.
                output = self.model(
                    input_ids, past_key_values=past, use_cache=True
                )
            last_logits = output.logits[:, -1, :]
            past = output.past_key_values

        seq._kv_cache = past

        if end >= seq.prompt_len:
            return self._sample_and_append(seq, last_logits[0])
        return None

    def run_decode(self, seq):
        """Process one decode step with stored KV cache.

        Feeds only the last generated token alongside the cached
        past_key_values from prefill. Updates the cache after the forward.
        """
        past = getattr(seq, "_kv_cache", None)
        input_ids = torch.tensor(
            [[seq.last_token]], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            if past is not None:
                output = self.model(input_ids, past_key_values=past, use_cache=True)
                seq._kv_cache = output.past_key_values
            else:
                output = self.model(input_ids, use_cache=False)
            logits = output.logits[:, -1, :]
        return self._sample_and_append(seq, logits[0])

    def run(self, scheduled_batch):
        """Execute a batch of ScheduledItems.

        Returns list of (seq, token_id) tuples for completed work units.
        """
        results = []
        for item in scheduled_batch.items:
            seq = item.seq
            if item.kind == "decode":
                if (
                    self.spec_decode_manager is not None
                    and seq.sampling_params.temperature == 0
                ):
                    emitted = self.spec_decode_manager.step(seq)
                    token_id = emitted[-1]
                else:
                    token_id = self.run_decode(seq)
                results.append((seq, token_id))
            elif item.kind == "prefill":
                token_id = self.run_prefill(seq, item.num_tokens)
                if token_id is not None:
                    results.append((seq, token_id))
        return results
