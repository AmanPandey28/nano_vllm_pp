"""Tutorial: KV cache generation — prefill/decode separation.

Prefill processes the full prompt once and stores the KV cache.
Decode feeds one token at a time using the stored cache.
This reduces O(n^2) to O(n) compute.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CacheLLM:
    def __init__(self, model_path: str, device: str = "cuda"):
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
    def generate(self, prompt: str, max_tokens: int = 32):
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
        return {"generated_text": gen_text}


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
    llm = CacheLLM(model_path)
    result = llm.generate("The capital of France is", max_tokens=20)
    print(result["generated_text"])
