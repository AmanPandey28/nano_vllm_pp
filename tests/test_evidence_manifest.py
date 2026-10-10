import sys
from types import SimpleNamespace

from nanovllm_pp.benchmarks import evidence


def test_manifest_has_portable_invocation_and_source_hashes(monkeypatch, tmp_path):
    model = tmp_path / "checkpoint"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["unused", str(model), "--repeats", "3"])
    monkeypatch.setattr(
        sys.modules["__main__"], "__spec__",
        SimpleNamespace(name="nanovllm_pp.benchmarks.bench_runtime_evidence"),
    )
    monkeypatch.setattr(evidence, "cuda_device_info", lambda: {"device": "test"})
    monkeypatch.setattr(evidence.subprocess, "check_output", lambda *a, **k: "test")
    manifest = evidence.environment_manifest(model)
    assert manifest["command"] == (
        "python -m nanovllm_pp.benchmarks.bench_runtime_evidence ./checkpoint --repeats 3"
    )
    assert str(tmp_path) not in manifest["command"]
    assert manifest["command_paths_normalized"]
    assert "tokenizer.json" in manifest["model_sha256"]
    assert "nanovllm_pp/engine/scheduler.py" in manifest["source_sha256"]
