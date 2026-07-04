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

    def __call__(self, logits, params):
        logits = self.apply_temperature(logits, params.temperature)
        if params.top_k:
            logits = self.apply_top_k(logits, params.top_k)
        if params.top_p < 1.0:
            logits = self.apply_top_p(logits, params.top_p)
        if params.temperature == 0:
            token_id = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            token_id = int(torch.multinomial(probs, num_samples=1).item())
        return token_id, logits


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

    def run_prefill(self, seq, num_tokens):
        """Process `num_tokens` tokens of the prompt.

        If this completes the full prompt, sample and append the first
        generated token. Stores past_key_values on the sequence for
        subsequent decode steps.
        """
        start = seq.num_computed_tokens
        end = min(start + num_tokens, len(seq.token_ids))
        token_slice = seq.token_ids[start:end]
        input_ids = torch.tensor(
            token_slice, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_ids, use_cache=True)
            last_logits = output.logits[:, -1, :]
            past = output.past_key_values

        seq._kv_cache = past

        if end >= seq.prompt_len:
            token_id, _ = self.sampler(last_logits[0], seq.sampling_params)
            seq.append_token(token_id)
            return token_id
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
        with torch.no_grad():
            if past is not None:
                output = self.model(input_ids, past_key_values=past, use_cache=True)
                seq._kv_cache = output.past_key_values
            else:
                output = self.model(input_ids, use_cache=False)
            logits = output.logits[:, -1, :]
        token_id, _ = self.sampler(logits[0], seq.sampling_params)
        seq.append_token(token_id)
        return token_id

    def run(self, scheduled_batch):
        """Execute a batch of ScheduledItems.

        Returns list of (seq, token_id) tuples for completed work units.
        """
        results = []
        for item in scheduled_batch.items:
            seq = item.seq
            if item.kind == "decode":
                token_id = self.run_decode(seq)
                results.append((seq, token_id))
            elif item.kind == "prefill":
                token_id = self.run_prefill(seq, item.num_tokens)
                if token_id is not None:
                    results.append((seq, token_id))
        return results
