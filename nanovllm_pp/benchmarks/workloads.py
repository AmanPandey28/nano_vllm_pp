"""Dataset adapters for reproducible text-generation serving benchmarks."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRequest:
    """One prompt plus the number of output tokens the benchmark requests."""

    prompt: str
    output_tokens: int
    source: str


@dataclass(frozen=True)
class BenchmarkWorkload:
    """Measured requests and optional warm-up requests for cache preparation."""

    name: str
    requests: tuple[BenchmarkRequest, ...]
    warmup_requests: tuple[BenchmarkRequest, ...] = ()
    metadata: dict = field(default_factory=dict)


_SYNTHETIC_SEEDS = (
    "Explain how a transformer inference runtime schedules requests and manages memory.",
    "Describe the relationship between attention, key value caching, and decoding.",
    "Compare latency, throughput, memory capacity, and numerical accuracy tradeoffs.",
    "A production serving system must coordinate models, schedulers, kernels, and APIs.",
)


def count_prompt_tokens(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=True))


def _fit_prompt(tokenizer, seed_text: str, target_tokens: int) -> str:
    """Build deterministic text close to an exact tokenizer-visible length."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("tokenizer produced no tokens for the synthetic seed")

    candidate_ids = (seed_ids * (target_tokens // len(seed_ids) + 2))[:target_tokens]
    prompt = tokenizer.decode(candidate_ids, skip_special_tokens=True)
    # Decode/encode boundaries can merge tokens. Adjust the candidate token
    # count a few times and retain the closest stable text if exact equality is
    # impossible for a particular tokenizer.
    best = prompt
    best_error = abs(count_prompt_tokens(tokenizer, prompt) - target_tokens)
    for _ in range(12):
        actual = count_prompt_tokens(tokenizer, prompt)
        if actual == target_tokens:
            return prompt
        adjustment = target_tokens - actual
        if adjustment > 0:
            candidate_ids.extend(
                (seed_ids * (adjustment // len(seed_ids) + 1))[:adjustment]
            )
        else:
            candidate_ids = candidate_ids[: max(1, len(candidate_ids) + adjustment)]
        prompt = tokenizer.decode(candidate_ids, skip_special_tokens=True)
        error = abs(count_prompt_tokens(tokenizer, prompt) - target_tokens)
        if error < best_error:
            best, best_error = prompt, error
    return best


def make_random_workload(
    tokenizer,
    *,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int,
) -> BenchmarkWorkload:
    """Create a fixed-shape synthetic workload, analogous to vLLM random data."""
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    requests = []
    for index in range(num_prompts):
        seed_text = (
            f"Request {seed + index}. {_SYNTHETIC_SEEDS[index % len(_SYNTHETIC_SEEDS)]}"
        )
        requests.append(
            BenchmarkRequest(
                prompt=_fit_prompt(tokenizer, seed_text, input_len),
                output_tokens=output_len,
                source="random",
            )
        )
    return BenchmarkWorkload(
        name=f"random-{input_len}x{output_len}",
        requests=tuple(requests),
        metadata={
            "requested_input_len": input_len,
            "requested_output_len": output_len,
            "seed": seed,
        },
    )


def _conversation_pair(record) -> tuple[str, str] | None:
    messages = record.get("conversations") or record.get("messages")
    if not isinstance(messages, list):
        return None
    aliases = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
    }
    previous_user = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = aliases.get(message.get("from") or message.get("role"))
        value = message.get("value")
        if value is None:
            value = message.get("content")
        if role == "user" and isinstance(value, str) and value.strip():
            previous_user = value.strip()
        elif (
            role == "assistant"
            and previous_user is not None
            and isinstance(value, str)
            and value.strip()
        ):
            return previous_user, value.strip()
    return None


def load_sharegpt_workload(
    tokenizer,
    dataset_path,
    *,
    num_prompts: int,
    seed: int,
    max_model_len: int,
    max_output_len: int,
    fixed_output_len: int | None = None,
) -> BenchmarkWorkload:
    """Load first-turn user/assistant pairs from common ShareGPT JSON files."""
    path = Path(dataset_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("conversations") or []
    if not isinstance(payload, list):
        raise TypeError("ShareGPT dataset must contain a JSON list")

    records = list(payload)
    rng = random.Random(seed)
    rng.shuffle(records)
    candidates = []
    scanned = 0
    for record in records:
        scanned += 1
        if not isinstance(record, dict):
            continue
        pair = _conversation_pair(record)
        if pair is None:
            continue
        prompt, answer = pair
        # Bounded tokenization avoids pathological conversations dominating
        # dataset preparation only to be rejected by max_model_len afterward.
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_model_len + 1,
        )
        input_tokens = len(prompt_ids)
        if fixed_output_len is not None:
            output_tokens = fixed_output_len
        else:
            output_tokens = len(
                tokenizer.encode(
                    answer,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_output_len,
                )
            )
        if input_tokens < 4 or output_tokens < 1:
            continue
        if input_tokens + output_tokens > max_model_len:
            continue
        candidates.append(
            BenchmarkRequest(
                prompt=prompt,
                output_tokens=output_tokens,
                source="sharegpt",
            )
        )
        if len(candidates) == num_prompts:
            break

    if len(candidates) < num_prompts:
        raise ValueError(
            f"ShareGPT filtering left {len(candidates)} requests, fewer than "
            f"--num-prompts={num_prompts}"
        )
    return BenchmarkWorkload(
        name="sharegpt",
        requests=tuple(candidates[:num_prompts]),
        metadata={
            "dataset_path": str(path),
            "selected_requests": len(candidates),
            "scanned_records": scanned,
            "seed": seed,
            "max_output_len": max_output_len,
            "fixed_output_len": fixed_output_len,
        },
    )


def common_prefix_tokens(tokenizer, prompts) -> int:
    token_lists = [
        tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts
    ]
    if not token_lists:
        return 0
    common = 0
    for values in zip(*token_lists):
        if len(set(values)) != 1:
            break
        common += 1
    return common


def make_prefix_repetition_workload(
    tokenizer,
    *,
    num_prompts: int,
    prefix_len: int,
    suffix_len: int,
    output_len: int,
    seed: int,
) -> BenchmarkWorkload:
    """Create requests sharing a long prefix and one cache-populating warm-up."""
    shared = _fit_prompt(
        tokenizer,
        (
            "System instruction: answer precisely and explain the engineering "
            "tradeoffs in transformer inference, scheduling, and memory management. "
        ),
        prefix_len,
    )
    requests = []
    for index in range(num_prompts):
        suffix = _fit_prompt(
            tokenizer,
            f"\n\nUnique request {seed + index}: analyze scenario {index}.",
            suffix_len,
        )
        requests.append(
            BenchmarkRequest(
                prompt=f"{shared}\n\n{suffix}",
                output_tokens=output_len,
                source="prefix_repetition",
            )
        )
    warm_suffix = _fit_prompt(
        tokenizer,
        "\n\nCache warm-up request: establish the shared prefix.",
        suffix_len,
    )
    warmup = BenchmarkRequest(
        prompt=f"{shared}\n\n{warm_suffix}",
        output_tokens=output_len,
        source="prefix_warmup",
    )
    actual_common = common_prefix_tokens(
        tokenizer, [warmup.prompt, *(request.prompt for request in requests)]
    )
    return BenchmarkWorkload(
        name=f"prefix-repetition-{prefix_len}+{suffix_len}",
        requests=tuple(requests),
        warmup_requests=(warmup,),
        metadata={
            "requested_prefix_len": prefix_len,
            "requested_suffix_len": suffix_len,
            "actual_common_prefix_tokens": actual_common,
            "requested_output_len": output_len,
            "seed": seed,
        },
    )


def workload_token_stats(tokenizer, workload: BenchmarkWorkload) -> dict:
    inputs = [
        count_prompt_tokens(tokenizer, request.prompt) for request in workload.requests
    ]
    outputs = [request.output_tokens for request in workload.requests]

    def describe(values):
        ordered = sorted(values)
        return {
            "min": min(values),
            "mean": round(sum(values) / len(values), 2),
            "median": ordered[len(ordered) // 2],
            "max": max(values),
            "total": sum(values),
        }

    return {"input_tokens": describe(inputs), "output_tokens": describe(outputs)}
