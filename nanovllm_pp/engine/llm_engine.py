import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..config import EngineConfig, ModelConfig
from ..engine.block_manager import BlockManager
from ..engine.model_runner import ModelRunner
from ..engine.scheduler import Scheduler
from ..engine.sequence import Sequence
from ..sampling_params import SamplingParams


class LLMEngine:
    """Core inference runtime: tokenization, scheduling, model execution.

    The engine owns the model, tokenizer, block manager, scheduler, and
    model runner. Its step() method is the main event loop — schedule,
    execute, postprocess — called repeatedly by generate() until all
    requests finish.

    Profiling can be enabled with enable_profiling() to get per-phase
    timing breakdowns and per-request latency metrics.
    """

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path

        engine_config = EngineConfig()
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        for k, v in kwargs.items():
            if hasattr(engine_config, k):
                setattr(engine_config, k, v)

        self.engine_config = engine_config

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        model_kwargs = {"trust_remote_code": True}
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs,
            )
            .to(self.device)
            .eval()
        )

        self.model_config = self._build_model_config(model_path)
        self.num_kv_blocks = self._estimate_kv_blocks()
        self.block_manager = BlockManager(self.num_kv_blocks, engine_config.block_size)
        self.model_runner = ModelRunner(
            self.model,
            self.model_config,
            engine_config,
            self.device,
            self.block_manager,
        )
        self.scheduler = Scheduler(engine_config, self.block_manager)
        self.step_counter = 0
        self._profiler = None
        self._request_tracker = None

    # ── profiling instrumentation ──────────────────────────────────

    def enable_profiling(self):
        from nanovllm_pp.observability.profiler import EngineProfiler, RequestTracker

        self._profiler = EngineProfiler()
        self._request_tracker = RequestTracker()
        return self

    def profile_report(self):
        if self._profiler:
            self._profiler.print_report()
        if self._request_tracker:
            self._request_tracker.print_report()

    # ── initialization helpers ─────────────────────────────────────

    def _build_model_config(self, model_path: str) -> ModelConfig:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        text_cfg = getattr(hf_config, "text_config", hf_config)

        return ModelConfig(
            model_path=model_path,
            num_layers=getattr(text_cfg, "num_hidden_layers", 0),
            num_kv_heads=getattr(
                text_cfg,
                "num_key_value_heads",
                getattr(text_cfg, "num_attention_heads", 0),
            ),
            num_attention_heads=text_cfg.num_attention_heads,
            head_dim=getattr(
                text_cfg,
                "head_dim",
                text_cfg.hidden_size // text_cfg.num_attention_heads,
            ),
            hidden_size=text_cfg.hidden_size,
            intermediate_size=getattr(
                text_cfg, "intermediate_size", getattr(text_cfg, "ffn_dim", 0)
            ),
            vocab_size=text_cfg.vocab_size,
            dtype=str(self.model.dtype),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
            rope_theta=getattr(text_cfg, "rope_theta", 10000.0),
        )

    def _estimate_kv_blocks(self) -> int:
        """Estimate how many KV cache blocks fit in remaining GPU memory.

        Formula:
          kv_per_token  = 2 * L * H_kv * d_head * sizeof(dtype)
          kv_per_block  = kv_per_token * block_size
          available     = total_gpu * utilization - model_weights - overhead
          num_blocks    = available / kv_per_block

        The factor of 2 accounts for both key and value tensors.
        """
        block_size = self.engine_config.block_size
        num_layers = self.model_config.num_layers
        num_kv_heads = self.model_config.num_kv_heads
        head_dim = self.model_config.head_dim
        dtype_bytes = 2

        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        kv_per_block = kv_per_token * block_size

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            available = total * self.engine_config.gpu_memory_utilization
            model_bytes = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            kv_budget = available - model_bytes - 512 * 1024 * 1024
            return max(10, int(kv_budget // kv_per_block))
        return 64

    # ── request lifecycle ──────────────────────────────────────────

    def add_request(self, prompt: str, sampling_params: SamplingParams):
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        seq = Sequence(token_ids, sampling_params)
        seq.enqueue_time = time.time()
        self.scheduler.add(seq)
        if self._request_tracker:
            self._request_tracker.record_enqueue(seq.request_id)
        return seq.request_id

    # ── main runtime loop ──────────────────────────────────────────

    def step(self):
        self.step_counter += 1

        if self._profiler:
            self._profiler.start_step(self.step_counter)
            self._profiler.tick("schedule")

        scheduled = self.scheduler.schedule()
        if self._profiler:
            self._profiler.tick("schedule")

        if not scheduled:
            return []

        model_kind = "prefill" if getattr(scheduled, "is_prefill", True) else "decode"
        if self._profiler:
            self._profiler.tick(model_kind)

        output_tokens = self.model_runner.run(scheduled)

        if self._profiler:
            self._profiler.tick(model_kind)
            self._profiler.tick("postprocess")

        finished = self.scheduler.postprocess(scheduled, output_tokens)

        if self._profiler:
            self._profiler.tick("postprocess")
            s = self.scheduler.debug_state()
            pf = sum(1 for i in getattr(scheduled, "items", []) if i.kind == "prefill")
            dc = sum(1 for i in getattr(scheduled, "items", []) if i.kind == "decode")
            pt = sum(
                i.num_tokens
                for i in getattr(scheduled, "items", [])
                if i.kind == "prefill"
            )
            self._profiler.end_step(
                num_prefill=pf,
                num_decode=dc,
                num_prefill_tokens=pt,
                free_blocks=s["blocks"]["free_blocks"],
                waiting=s["waiting"],
                running=s["running"],
            )

        if self._request_tracker:
            for seq in finished:
                self._request_tracker.record_completion(
                    seq.request_id, seq.num_generated_tokens
                )
            for seq in list(self.scheduler.running):
                if seq.num_generated_tokens >= 1:
                    try:
                        self._request_tracker.record_first_token(seq.request_id)
                    except Exception:
                        pass

        return finished

    def generate(self, prompts, sampling_params):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        request_id_to_seq = {}
        for prompt, sp in zip(prompts, sampling_params):
            rid = self.add_request(prompt, sp)
            request_id_to_seq[rid] = None

        finished_seqs = []
        while self.scheduler.has_unfinished():
            batch_finished = self.step()
            for seq in batch_finished:
                seq.completion_time = time.time()
                finished_seqs.append(seq)
                request_id_to_seq[seq.request_id] = seq

        outputs = []
        for prompt in prompts:
            encoded = self.tokenizer.encode(prompt, add_special_tokens=True)
            prompt_len = len(encoded)
            matched = None
            for seq in finished_seqs:
                if seq.token_ids[:prompt_len] == encoded:
                    matched = seq
                    break
            if matched is None:
                continue
            outputs.append(
                {
                    "text": self.tokenizer.decode(
                        matched.token_ids, skip_special_tokens=True
                    ),
                    "generated_text": self.tokenizer.decode(
                        matched.token_ids[prompt_len:], skip_special_tokens=True
                    ),
                    "token_ids": matched.token_ids,
                    "request_id": matched.request_id,
                    "num_prompt_tokens": matched.prompt_len,
                    "num_completion_tokens": matched.num_generated_tokens,
                }
            )

        return outputs
