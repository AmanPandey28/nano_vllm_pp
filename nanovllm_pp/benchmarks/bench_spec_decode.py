"""Benchmark: n-gram speculative decoding vs standard greedy decode.

Uses a repetitive prompt to maximize n-gram match rate and measures
acceptance rate, target forwards saved, and wall-clock speedup.
"""

import time
import sys
import torch

sys.path.insert(0, ".")

from nanovllm_pp.spec_decode.manager import SpecDecodeManager


class SpecBenchRunner:
    def __init__(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        from nanovllm_pp.engine.model_runner import ModelRunner
        from nanovllm_pp.config import ModelConfig, EngineConfig

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

    def run_standard(self, prompt, max_tokens):
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        start = time.time()
        for _ in range(max_tokens):
            with torch.no_grad():
                out = self.model(ids)
                logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        elapsed = time.time() - start
        return {
            "method": "standard",
            "elapsed_s": round(elapsed, 3),
            "target_forwards": max_tokens,
            "tokens_generated": max_tokens,
        }

    def run_spec_decode(self, prompt, max_tokens, max_spec=4):
        from nanovllm_pp.engine.sequence import Sequence
        from nanovllm_pp.sampling_params import SamplingParams

        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        seq = Sequence(ids, sp)
        seq.num_computed_tokens = len(ids)

        manager = SpecDecodeManager(self.model_runner, max_spec_tokens=max_spec)
        start = time.time()
        generated = 0
        while generated < max_tokens:
            n = manager.step(seq)
            if n <= 0:
                break
            generated += n
            if generated >= max_tokens:
                break
        elapsed = time.time() - start
        return {
            "method": "spec_decode",
            "elapsed_s": round(elapsed, 3),
            "tokens_generated": generated,
            "acceptance_rate": manager.metrics.acceptance_rate,
            "tokens_proposed": manager.metrics.tokens_proposed,
            "tokens_accepted": manager.metrics.tokens_accepted,
            "target_forwards": manager.metrics.target_forwards,
            "empty_proposals": manager.metrics.proposals_empty,
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
    std = runner.run_standard(prompt, max_tokens)
    spec = runner.run_spec_decode(prompt, max_tokens)

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
