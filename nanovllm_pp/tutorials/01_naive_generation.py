"""Tutorial: naive generation baseline — O(n^2) autoregressive decoding.

Demonstrates why naive decoding is inefficient: each step recomputes the
full accumulated sequence, resulting in O(n^2) total compute.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm_pp.device import model_dtype_kwargs, resolve_device, resolve_dtype


class TinyLLM:
    def __init__(self, model_path: str, device: str = "cuda"):
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
    def generate(self, prompt: str, max_tokens: int = 32):
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = ids.shape[1]
        for _ in range(max_tokens):
            out = self.model(ids)
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        gen_text = self.tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
        return {"generated_text": gen_text}


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
    llm = TinyLLM(model_path)
    result = llm.generate("The capital of France is", max_tokens=20)
    print(result["generated_text"])
