"""Check that public summaries agree with the underlying measured trials."""

import csv
import json
from pathlib import Path

import pytest

from nanovllm_pp.benchmarks.bench_runtime_evidence import summarize_case

EVIDENCE = Path(__file__).resolve().parents[1] / "benchmark_results/runtime_evidence"


def test_csv_and_case_artifacts_agree_with_raw_trials():
    aggregate = json.loads((EVIDENCE / "results.json").read_text())
    with (EVIDENCE / "summary.csv").open(newline="") as stream:
        rows = {row["case"]: row for row in csv.DictReader(stream)}
    assert rows.keys() == aggregate["cases"].keys()
    for name, case in aggregate["cases"].items():
        individual = json.loads((EVIDENCE / f"{name}.json").read_text())
        assert individual["trials"] == case["trials"]
        assert case["summary"] == summarize_case(name, case["trials"])
        for key, value in case["summary"].items():
            if isinstance(value, (int, float)):
                assert float(rows[name][key]) == pytest.approx(value)
        for trial in case["trials"]:
            assert trial["final_block_state"]["used_blocks"] == 0
            assert trial["successful_requests"] == len(trial["requests"])
            assert trial["total_output_tokens"] == sum(
                len(request["generated_token_ids"]) for request in trial["requests"]
            )


def test_readme_serving_values_come_from_the_artifacts():
    readme = (EVIDENCE.parents[1] / "README.md").read_text()
    aggregate = json.loads((EVIDENCE / "results.json").read_text())
    for case in aggregate["cases"].values():
        assert f"{case['summary']['output_tps']:.3f}" in readme
