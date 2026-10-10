import json

import pytest

from nanovllm_pp.benchmarks.bench_serving import required_blocks
from nanovllm_pp.benchmarks.serving_harness import (
    RequestRecord,
    latency_summary,
    percentile,
    poisson_arrival_offsets,
    request_metrics,
)
from nanovllm_pp.benchmarks.workloads import (
    common_prefix_tokens,
    load_sharegpt_workload,
    make_prefix_repetition_workload,
    make_random_workload,
    workload_token_stats,
)


class WordTokenizer:
    def __init__(self):
        self.words = {}
        self.reverse = {}

    def encode(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        values = []
        for word in text.split():
            if word not in self.words:
                token_id = len(self.words) + 1
                self.words[word] = token_id
                self.reverse[token_id] = word
            values.append(self.words[word])
        encoded = ([0] if add_special_tokens else []) + values
        return encoded[:max_length] if truncation and max_length else encoded

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(
            self.reverse[token_id]
            for token_id in token_ids
            if token_id != 0 or not skip_special_tokens
        )


def test_random_workload_has_reproducible_requested_shapes():
    tokenizer = WordTokenizer()
    workload = make_random_workload(
        tokenizer,
        num_prompts=3,
        input_len=12,
        output_len=5,
        seed=9,
    )
    stats = workload_token_stats(tokenizer, workload)

    assert len(workload.requests) == 3
    assert stats["input_tokens"]["min"] == 12
    assert stats["input_tokens"]["max"] == 12
    assert stats["output_tokens"]["total"] == 15


def test_sharegpt_loader_accepts_from_value_and_role_content(tmp_path):
    dataset = [
        {
            "conversations": [
                {"from": "human", "value": "one two three four"},
                {"from": "gpt", "value": "five six seven"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "alpha beta gamma delta"},
                {"role": "assistant", "content": "epsilon zeta eta"},
            ]
        },
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    workload = load_sharegpt_workload(
        WordTokenizer(),
        path,
        num_prompts=2,
        seed=3,
        max_model_len=32,
        max_output_len=2,
    )

    assert len(workload.requests) == 2
    assert {request.output_tokens for request in workload.requests} == {2}
    assert workload.metadata["selected_requests"] == 2


def test_prefix_workload_reports_a_real_shared_token_prefix():
    tokenizer = WordTokenizer()
    workload = make_prefix_repetition_workload(
        tokenizer,
        num_prompts=4,
        prefix_len=16,
        suffix_len=5,
        output_len=3,
        seed=7,
    )
    prompts = [
        workload.warmup_requests[0].prompt,
        *(request.prompt for request in workload.requests),
    ]

    assert workload.metadata["actual_common_prefix_tokens"] >= 15
    assert (
        common_prefix_tokens(tokenizer, prompts)
        == workload.metadata["actual_common_prefix_tokens"]
    )


def test_required_blocks_uses_largest_closed_concurrency_set():
    tokenizer = WordTokenizer()
    workload = make_random_workload(
        tokenizer,
        num_prompts=4,
        input_len=9,
        output_len=3,
        seed=1,
    )
    assert required_blocks(tokenizer, workload, block_size=4, concurrency=2) == 6


def test_percentiles_and_request_metrics_include_itl_and_prefix_hits():
    records = [
        RequestRecord(
            index=0,
            source="test",
            scheduled_time=0.0,
            send_time=0.1,
            prompt_tokens=10,
            requested_output_tokens=3,
            request_id="a",
            cached_prompt_tokens=4,
            token_times=[0.2, 0.3, 0.5],
            completion_time=0.5,
        ),
        RequestRecord(
            index=1,
            source="test",
            scheduled_time=0.0,
            send_time=0.2,
            prompt_tokens=6,
            requested_output_tokens=2,
            request_id="b",
            token_times=[0.3, 0.5],
            completion_time=0.5,
        ),
    ]

    metrics = request_metrics(records, duration_s=0.4)

    assert percentile([0, 10], 50) == 5
    assert latency_summary([10, 20, 30])["p95"] == pytest.approx(29)
    assert metrics["request_throughput_rps"] == 5
    assert metrics["output_token_throughput_tps"] == 12.5
    assert metrics["cached_prompt_tokens"] == 4
    assert metrics["prefix_cache_hit_fraction"] == 0.25
    assert metrics["prefix_hit_requests"] == 1
    assert metrics["prefix_request_hit_fraction"] == 0.5
    assert metrics["prefill_tokens_computed"] == 12
    assert metrics["processed_token_throughput_tps"] == 37.5
    assert metrics["latency_ms"]["itl"]["p50"] == pytest.approx(200)


def test_poisson_offsets_are_seeded_and_increasing():
    first = poisson_arrival_offsets(5, request_rate=4.0, seed=11)
    second = poisson_arrival_offsets(5, request_rate=4.0, seed=11)

    assert first == second
    assert first[0] == 0
    assert first == sorted(first)
    assert poisson_arrival_offsets(3, request_rate=float("inf"), seed=1) == [
        0.0,
        0.0,
        0.0,
    ]
