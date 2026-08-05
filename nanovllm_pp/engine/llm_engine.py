import time
from dataclasses import fields

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import EngineConfig, ModelConfig
from ..engine.block_manager import BlockManager
from ..engine.model_runner import ModelRunner
from ..engine.scheduler import Scheduler
from ..engine.sequence import Sequence, SequenceStatus
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

        load_in_4bit = kwargs.pop("load_in_4bit", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive")
        config_names = {field.name for field in fields(EngineConfig)}
        unknown = sorted(set(kwargs) - config_names)
        if unknown:
            raise TypeError(f"unknown engine option(s): {', '.join(unknown)}")
        engine_config = EngineConfig(**kwargs)
        if not engine_config.enforce_eager:
            raise NotImplementedError(
                "this educational runner currently supports eager execution only"
            )
        if engine_config.enable_prefix_caching:
            raise NotImplementedError(
                "prefix hashes are implemented in BlockManager, but physical "
                "Hugging Face KV reuse is not integrated; keep prefix caching disabled"
            )

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
            model_kwargs["device_map"] = {"": 0}
        elif load_in_8bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        if (load_in_4bit or load_in_8bit) and self.device != "cuda":
            raise RuntimeError(
                "bitsandbytes 4/8-bit loading is supported only on CUDA in "
                "this project; use the educational weight-only modules for CPU"
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        # Accelerate/bitsandbytes dispatches quantized modules while loading;
        # calling .to() afterward is unsupported for several bnb versions.
        if not (load_in_4bit or load_in_8bit):
            self.model = self.model.to(self.device)
        self.model.eval()

        self.model_config = self._build_model_config(model_path)
        self.num_kv_blocks = self._estimate_kv_blocks()
        self.block_manager = BlockManager(
            self.num_kv_blocks,
            engine_config.block_size,
            enable_prefix_caching=engine_config.enable_prefix_caching,
        )
        self.model_runner = ModelRunner(
            self.model,
            self.model_config,
            engine_config,
            self.device,
            self.block_manager,
        )
        self.scheduler = Scheduler(engine_config, self.block_manager)
        if engine_config.enable_spec_decode:
            from ..spec_decode.manager import SpecDecodeManager

            self.model_runner.spec_decode_manager = SpecDecodeManager(
                self.model_runner, max_spec_tokens=engine_config.max_spec_tokens
            )
        self.step_counter = 0
        self._profiler = None
        self._request_tracker = None
        self._trace = None
        self._sequences = {}

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

    def enable_tracing(self):
        """Record structured in-memory lifecycle events for later JSONL export."""
        from nanovllm_pp.observability.trace import EngineTrace

        self._trace = EngineTrace()
        return self

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
          headroom      = min(driver_free,
                              total_gpu * utilization - torch_allocated)
          num_blocks    = (headroom - runtime_reserve) / kv_per_block

        The factor of 2 accounts for both key and value tensors.
        """
        block_size = self.engine_config.block_size
        num_layers = self.model_config.num_layers
        num_kv_heads = self.model_config.num_kv_heads
        head_dim = self.model_config.head_dim
        dtype_bytes = next(self.model.parameters()).element_size()

        kv_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        kv_per_block = kv_per_token * block_size
        if kv_per_block <= 0:
            raise ValueError(
                "model configuration is missing layer/head dimensions needed "
                "to estimate KV capacity"
            )

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            target_headroom = (
                total_bytes * self.engine_config.gpu_memory_utilization
                - torch.cuda.memory_allocated()
            )
            # Keep headroom for activations, allocator fragmentation, and
            # non-PyTorch CUDA allocations. This remains an admission estimate;
            # Hugging Face owns the actual dynamic cache tensors.
            kv_budget = min(free_bytes, target_headroom) - 512 * 1024 * 1024
            num_blocks = int(kv_budget // kv_per_block)
            if num_blocks < 1:
                raise RuntimeError(
                    "insufficient CUDA headroom for one logical KV block; lower "
                    "block_size/model size or increase gpu_memory_utilization"
                )
            return num_blocks
        return 64

    # ── request lifecycle ──────────────────────────────────────────

    def add_request(self, prompt: str, sampling_params: SamplingParams):
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be a SamplingParams instance")
        if sampling_params.prompt_logprobs:
            raise NotImplementedError("prompt_logprobs are not implemented")
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not token_ids:
            fallback_token = self.tokenizer.bos_token_id
            if fallback_token is None:
                fallback_token = self.tokenizer.eos_token_id
            if fallback_token is None:
                raise ValueError("tokenizer produced no tokens for the empty prompt")
            token_ids = [fallback_token]
        if len(token_ids) + sampling_params.max_tokens > self.engine_config.max_model_len:
            raise ValueError(
                "prompt tokens + max_tokens exceed max_model_len: "
                f"{len(token_ids)} + {sampling_params.max_tokens} > "
                f"{self.engine_config.max_model_len}"
            )
        if (
            not self.engine_config.enable_chunked_prefill
            and len(token_ids) > self.engine_config.max_num_batched_tokens
        ):
            raise ValueError(
                "prompt exceeds max_num_batched_tokens; enable chunked prefill "
                "or increase the token budget"
            )
        seq = Sequence(token_ids, sampling_params)
        seq.prompt = prompt
        seq.eos_token_id = self.tokenizer.eos_token_id
        seq.enqueue_time = time.perf_counter()
        self.scheduler.add(seq)
        self._sequences[seq.request_id] = seq
        if self._request_tracker:
            self._request_tracker.record_enqueue(seq.request_id)
        if self._trace:
            self._trace.add(
                step=self.step_counter,
                event_type="enqueue",
                mode="waiting",
                seq_ids=[seq.seq_id],
                num_tokens=seq.prompt_len,
            )
        return seq.request_id

    def get_request(self, request_id: str):
        return self._sequences.get(request_id)

    def is_request_finished(self, request_id: str) -> bool:
        seq = self.get_request(request_id)
        return seq is not None and seq.status == SequenceStatus.FINISHED

    def forget_request(self, request_id: str):
        """Release retained token/output state after a caller consumes it."""
        seq = self._sequences.get(request_id)
        if seq is None:
            return
        if seq.status != SequenceStatus.FINISHED:
            raise RuntimeError("cannot forget an unfinished request")
        del self._sequences[request_id]

    def _mark_stop_strings(self, scheduled):
        for seq in scheduled.seqs:
            if not seq.sampling_params.stop or not seq.generated_token_ids:
                continue
            text = self.tokenizer.decode(
                seq.generated_token_ids, skip_special_tokens=False
            )
            matches = [stop for stop in seq.sampling_params.stop if stop in text]
            if matches:
                seq.stop_matched = min(matches, key=text.find)

    # ── main runtime loop ──────────────────────────────────────────

    def step(self):
        step_start = time.perf_counter()
        self.step_counter += 1

        if self._profiler:
            self._profiler.start_step(self.step_counter)
            self._profiler.tick("schedule")

        scheduled = self.scheduler.schedule()
        if self._profiler:
            self._profiler.tick("schedule")

        if not scheduled:
            state = self.scheduler.debug_state()
            if self.scheduler.has_unfinished():
                raise RuntimeError(
                    "scheduler made no progress with unfinished requests; "
                    f"state={state}. Check token budget and KV block capacity."
                )
            return []

        if scheduled.is_prefill:
            model_kind = "prefill"
        elif scheduled.is_decode:
            model_kind = "decode"
        else:
            model_kind = "mixed"
        if self._trace:
            block_state = self.block_manager.stats()
            self._trace.add(
                step=self.step_counter,
                event_type="schedule",
                mode=model_kind,
                seq_ids=[seq.seq_id for seq in scheduled.seqs],
                num_tokens=sum(item.num_tokens for item in scheduled.items),
                free_blocks=block_state["free_blocks"],
                used_blocks=block_state["used_blocks"],
            )
        if self._profiler:
            self._profiler.tick(model_kind)

        output_tokens = self.model_runner.run(scheduled)
        self._mark_stop_strings(scheduled)

        if self._profiler:
            self._profiler.tick(model_kind)
            self._profiler.tick("postprocess")

        finished = self.scheduler.postprocess(scheduled, output_tokens)

        now = time.perf_counter()
        for seq in scheduled.seqs:
            if seq.num_generated_tokens >= 1 and seq.first_token_time is None:
                seq.first_token_time = now
                if self._request_tracker:
                    self._request_tracker.record_first_token(seq.request_id)
        for seq in finished:
            seq.completion_time = now
            if self._request_tracker:
                self._request_tracker.record_completion(
                    seq.request_id, seq.num_generated_tokens
                )
            if self._trace:
                self._trace.add(
                    step=self.step_counter,
                    event_type="finish",
                    mode=seq.finish_reason,
                    seq_ids=[seq.seq_id],
                    num_tokens=seq.num_generated_tokens,
                    latency_ms=(now - seq.enqueue_time) * 1000,
                )

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

        if self._trace:
            block_state = self.block_manager.stats()
            self._trace.add(
                step=self.step_counter,
                event_type="step_complete",
                mode=model_kind,
                seq_ids=[seq.seq_id for seq in scheduled.seqs],
                num_tokens=sum(item.num_tokens for item in scheduled.items),
                free_blocks=block_state["free_blocks"],
                used_blocks=block_state["used_blocks"],
                latency_ms=(time.perf_counter() - step_start) * 1000,
            )

        return finished

    def generate(self, prompts, sampling_params):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        else:
            sampling_params = list(sampling_params)
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params must contain one item per prompt")

        request_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            rid = self.add_request(prompt, sp)
            request_ids.append(rid)

        while self.scheduler.has_unfinished():
            self.step()

        outputs = [self.get_output(request_id) for request_id in request_ids]
        for request_id in request_ids:
            self.forget_request(request_id)
        return outputs

    def get_output(self, request_id: str):
        seq = self.get_request(request_id)
        if seq is None:
            raise KeyError(f"unknown request_id: {request_id}")
        generated_text = self.tokenizer.decode(
            seq.generated_token_ids, skip_special_tokens=True
        )
        if seq.stop_matched and seq.stop_matched in generated_text:
            generated_text = generated_text.split(seq.stop_matched, 1)[0]
        return {
            "text": seq.prompt + generated_text,
            "generated_text": generated_text,
            "token_ids": list(seq.token_ids),
            "request_id": seq.request_id,
            "num_prompt_tokens": seq.prompt_len,
            "num_completion_tokens": seq.num_generated_tokens,
            "finish_reason": seq.finish_reason,
            "logprobs": (
                list(seq.generated_logprobs) if seq.sampling_params.logprobs else None
            ),
        }
