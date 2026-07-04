"""Benchmark: naive generation vs KV cache vs NanoVLLM++ engine.

Compares three implementations on identical prompts to measure the
speedup from KV caching and scheduling overhead.
"""

import time
import sys
import torch

sys.path.insert(0, ".")

from nanovllm_pp import LLM, SamplingParams


class NaiveLLM:
    """Minimal LLM — recomputes full context at every step. O(n^2) compute."""

    def __init__(self, model_path, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device if torch.cuda.is_available() else "cpu"
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

    @torch.inference_mode()
    def generate(self, prompt, max_tokens=32):
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = ids.shape[1]
        for _ in range(max_tokens):
            out = self.model(ids)
            next_id = torch.argmax(out.logits[:, -1], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        gen_text = self.tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
        return {"generated_text": gen_text, "tokens": ids.shape[1] - prompt_len}


class CacheLLM:
    """LLM with prefill/decode separation using past_key_values."""

    def __init__(self, model_path, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device if torch.cuda.is_available() else "cpu"
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

    @torch.inference_mode()
    def generate(self, prompt, max_tokens=32):
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = ids.shape[1]
        out = self.model(ids, use_cache=True)
        logits = out.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        past = out.past_key_values
        for _ in range(max_tokens - 1):
            out = self.model(next_id, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            past = out.past_key_values
        gen_text = self.tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
        return {"generated_text": gen_text, "tokens": ids.shape[1] - prompt_len}


def run_naive(model_path, prompts, max_tokens):
    llm = NaiveLLM(model_path)
    start = time.time()
    total_tokens = sum(
        llm.generate(p, max_tokens=max_tokens)["tokens"] for p in prompts
    )
    elapsed = time.time() - start
    return {
        "method": "naive",
        "elapsed_s": round(elapsed, 3),
        "requests": len(prompts),
        "tok/s": round(total_tokens / elapsed, 1),
    }


def run_cache(model_path, prompts, max_tokens):
    llm = CacheLLM(model_path)
    start = time.time()
    total_tokens = sum(
        llm.generate(p, max_tokens=max_tokens)["tokens"] for p in prompts
    )
    elapsed = time.time() - start
    return {
        "method": "kv_cache",
        "elapsed_s": round(elapsed, 3),
        "requests": len(prompts),
        "tok/s": round(total_tokens / elapsed, 1),
    }


def run_engine(model_path, prompts, max_tokens):
    llm = LLM(model_path, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    start = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - start
    total_tokens = sum(o["num_completion_tokens"] for o in outputs)
    return {
        "method": "nano_vllm_pp",
        "elapsed_s": round(elapsed, 3),
        "requests": len(prompts),
        "tok/s": round(total_tokens / elapsed, 1),
        "engine_steps": llm.step_counter,
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "Python is a programming language",
    ]
    max_tokens = 16

    print(f"Benchmark: {model_path}")
    print(f"  Prompts: {len(prompts)}, max_tokens: {max_tokens}\n")
    print(f"  {'Method':<20} {'Time(s)':<10} {'tok/s':<10} {'Extra'}")

    for runner in [run_naive, run_cache, run_engine]:
        result = runner(model_path, prompts, max_tokens)
        extra = f"steps={result['engine_steps']}" if "engine_steps" in result else ""
        print(
            f"  {result['method']:<20} {result['elapsed_s']:<10} "
            f"{result['tok/s']:<10} {extra}"
        )
