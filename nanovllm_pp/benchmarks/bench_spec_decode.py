"""Benchmark: n-gram speculative decoding vs cached greedy decode.

Uses a repetitive prompt to maximize n-gram match rate and measures
acceptance rate, target forwards saved, and wall-clock speedup. Both paths
use the same Hugging Face KV cache; this is not a comparison against an
artificial full-recomputation baseline.
"""

import statistics
import sys
import time

import torch

sys.path.insert(0, ".")

from nanovllm_pp.device import (
    model_dtype_kwargs,
    resolve_device,
    resolve_dtype,
    synchronize,
)
from nanovllm_pp.spec_decode.manager import SpecDecodeManager


class SpecBenchRunner:
    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = str(resolve_device("cuda"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        dtype = resolve_dtype(torch.device(self.device), "auto")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **model_dtype_kwargs(dtype),
            )
            .to(self.device)
            .eval()
        )

        from nanovllm_pp.config import EngineConfig, ModelConfig
        from nanovllm_pp.engine.model_runner import ModelRunner

        cfg = ModelConfig(
            vocab_size=self.model.config.vocab_size,
            num_layers=1,
            num_kv_heads=1,
            num_attention_heads=1,
            head_dim=1,
            hidden_size=1,
            intermediate_size=1,
        )
        ec = EngineConfig()
        self.model_runner = ModelRunner(self.model, cfg, ec, device=self.device)

    def _sequence(self, prompt, max_tokens):
        from nanovllm_pp.engine.sequence import Sequence
        from nanovllm_pp.sampling_params import SamplingParams

        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        seq = Sequence(ids, params)
        seq.eos_token_id = self.tokenizer.eos_token_id
        return seq

    def _synchronize(self):
        synchronize(self.device)

    def run_standard(self, prompt, max_tokens):
        seq = self._sequence(prompt, max_tokens)
        self._synchronize()
        start = time.perf_counter()
        self.model_runner.run_prefill(seq, seq.prompt_len)
        while seq.num_generated_tokens < max_tokens:
            self.model_runner.run_decode(seq)
        self._synchronize()
        elapsed = time.perf_counter() - start
        return {
            "method": "cached_greedy",
            "elapsed_s": round(elapsed, 3),
            "target_forwards": max_tokens,
            "tokens_generated": seq.num_generated_tokens,
            "generated_token_ids": list(seq.generated_token_ids),
        }

    def run_spec_decode(self, prompt, max_tokens, max_spec=4):
        seq = self._sequence(prompt, max_tokens)
        manager = SpecDecodeManager(self.model_runner, max_spec_tokens=max_spec)
        self._synchronize()
        start = time.perf_counter()
        self.model_runner.run_prefill(seq, seq.prompt_len)
        while seq.num_generated_tokens < max_tokens:
            manager.step(seq)
        self._synchronize()
        elapsed = time.perf_counter() - start
        return {
            "method": "spec_decode",
            "elapsed_s": round(elapsed, 3),
            "tokens_generated": seq.num_generated_tokens,
            "acceptance_rate": manager.metrics.acceptance_rate,
            "tokens_proposed": manager.metrics.tokens_proposed,
            "tokens_accepted": manager.metrics.tokens_accepted,
            # One prefill forward plus the measured speculative decode passes.
            "target_forwards": 1 + manager.metrics.target_forwards,
            "empty_proposals": manager.metrics.proposals_empty,
            "generated_token_ids": list(seq.generated_token_ids),
        }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"

    prompt = (
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome. The capital of Spain is Madrid. "
        "The capital of France is"
    )
    max_tokens = 12
    print(f"Model: {model_path}")
    print(f"Max tokens: {max_tokens}\n")

    runner = SpecBenchRunner(model_path)
    runner.run_standard("CUDA warm-up", 2)
    standard_samples = []
    speculative_samples = []
    std = None
    spec = None
    for _ in range(3):
        std = runner.run_standard(prompt, max_tokens)
        spec = runner.run_spec_decode(prompt, max_tokens)
        standard_samples.append(std["elapsed_s"])
        speculative_samples.append(spec["elapsed_s"])
        if spec["generated_token_ids"] != std["generated_token_ids"]:
            raise RuntimeError("speculative decoding changed greedy output tokens")
    std["elapsed_s"] = round(statistics.median(standard_samples), 3)
    spec["elapsed_s"] = round(statistics.median(speculative_samples), 3)

    print(f"  {'Method':<20} {'Time(s)':<10} {'Fwds':<8} {'Accept':<8}")
    print(f"  {std['method']:<20} {std['elapsed_s']:<10} {std['target_forwards']:<8} —")
    print(
        f"  {spec['method']:<20} {spec['elapsed_s']:<10} "
        f"{spec['target_forwards']:<8} {spec['acceptance_rate']:.2f}"
    )
    print(
        f"\n  Proposed: {spec['tokens_proposed']}, "
        f"Accepted: {spec['tokens_accepted']}, "
        f"Empty proposals: {spec['empty_proposals']}"
    )
    print(f"  Cached samples (s): {standard_samples}")
    print(f"  Spec samples (s):   {speculative_samples}")
    print("  Greedy token parity: PASS")
