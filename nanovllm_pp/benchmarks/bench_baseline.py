"""CUDA benchmark: naive generation vs KV cache vs NanoVLLM++ engine.

Compares three implementations on identical prompts to measure the
speedup from KV caching and scheduling overhead.
"""

import gc
import sys

import torch

sys.path.insert(0, ".")

from nanovllm_pp import LLM, SamplingParams
from nanovllm_pp.device import (
    cuda_device_info,
    model_dtype_kwargs,
    resolve_device,
    resolve_dtype,
    timed_call,
)


class NaiveLLM:
    """Minimal LLM — recomputes full context at every step. O(n^2) compute."""

    def __init__(self, model_path, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = str(resolve_device(device))
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

    @torch.inference_mode()
    def generate(self, prompt, max_tokens=32):
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = ids.shape[1]
        for _ in range(max_tokens):
            out = self.model(ids)
            next_id = torch.argmax(out.logits[:, -1], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        gen_text = self.tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
        return {
            "generated_text": gen_text,
            "generated_token_ids": ids[0, prompt_len:].tolist(),
            "tokens": ids.shape[1] - prompt_len,
        }


class CacheLLM:
    """LLM with prefill/decode separation using past_key_values."""

    def __init__(self, model_path, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = str(resolve_device(device))
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
        return {
            "generated_text": gen_text,
            "generated_token_ids": ids[0, prompt_len:].tolist(),
            "tokens": ids.shape[1] - prompt_len,
        }


def _release_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


def _measure_generator(llm, prompts, max_tokens):
    llm.generate("CUDA warm-up", max_tokens=2)

    def generate_all():
        return [llm.generate(prompt, max_tokens=max_tokens) for prompt in prompts]

    results, elapsed = timed_call(generate_all, device=llm.device)
    total_tokens = sum(result["tokens"] for result in results)
    token_ids = [result["generated_token_ids"] for result in results]
    return total_tokens, token_ids, elapsed


def run_naive(model_path, prompts, max_tokens, device):
    _release_cuda_cache()
    llm = NaiveLLM(model_path, device=str(device))
    total_tokens, token_ids, elapsed = _measure_generator(llm, prompts, max_tokens)
    return {
        "method": "naive",
        "elapsed_s": round(elapsed, 3),
        "requests": len(prompts),
        "tok/s": round(total_tokens / elapsed, 1),
        "generated_token_ids": token_ids,
    }


def run_cache(model_path, prompts, max_tokens, device):
    _release_cuda_cache()
    llm = CacheLLM(model_path, device=str(device))
    total_tokens, token_ids, elapsed = _measure_generator(llm, prompts, max_tokens)
    return {
        "method": "kv_cache",
        "elapsed_s": round(elapsed, 3),
        "requests": len(prompts),
        "tok/s": round(total_tokens / elapsed, 1),
        "generated_token_ids": token_ids,
    }


def run_engine(model_path, prompts, max_tokens, device):
    _release_cuda_cache()
    llm = LLM(model_path, enforce_eager=True, device=str(device))
    llm.generate(["CUDA warm-up"], SamplingParams(temperature=0.0, max_tokens=2))
    steps_before = llm.step_counter
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    outputs, elapsed = timed_call(llm.generate, prompts, sp, device=llm.device)
    total_tokens = sum(o["num_completion_tokens"] for o in outputs)
    return {
        "method": "nano_vllm_pp",
        "elapsed_s": round(elapsed, 3),
        "requests": len(prompts),
        "tok/s": round(total_tokens / elapsed, 1),
        "engine_steps": llm.step_counter - steps_before,
        "generated_token_ids": [
            output["token_ids"][output["num_prompt_tokens"] :] for output in outputs
        ],
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
    device = resolve_device("cuda")
    print(f"  CUDA: {cuda_device_info(device)}")
    print(f"  Prompts: {len(prompts)}, max_tokens: {max_tokens}\n")
    print(f"  {'Method':<20} {'Time(s)':<10} {'tok/s':<10} {'Extra'}")

    results = []
    for runner in [run_naive, run_cache, run_engine]:
        result = runner(model_path, prompts, max_tokens, device)
        results.append(result)
        extra = f"steps={result['engine_steps']}" if "engine_steps" in result else ""
        print(
            f"  {result['method']:<20} {result['elapsed_s']:<10} "
            f"{result['tok/s']:<10} {extra}"
        )
    reference = results[0]["generated_token_ids"]
    if any(result["generated_token_ids"] != reference for result in results[1:]):
        raise RuntimeError("greedy token parity failed across benchmark methods")
    print("\n  Greedy token parity: PASS")
