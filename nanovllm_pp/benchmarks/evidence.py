"""Environment manifests shared by reproducible runtime measurements."""

import hashlib
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from nanovllm_pp.device import cuda_device_info


def environment_manifest(model_path):
    versions = {}
    for package in ("torch", "transformers", "triton", "bitsandbytes", "accelerate"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    hashes = {}
    path = Path(model_path)
    for filename in (
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "generation_config.json", "special_tokens_map.json",
    ):
        file = path / filename
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[filename] = digest.hexdigest()
    commands = {
        "git_revision": ["git", "rev-parse", "HEAD"],
        "git_status": [
            "git",
            "status",
            "--short",
            "--",
            "nanovllm_pp/engine",
            "nanovllm_pp/paged",
            "nanovllm_pp/benchmarks",
            "nanovllm_pp/server",
            "nanovllm_pp/observability",
            "tests",
            "docs",
            "README.md",
        ],
        "gpu_state": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit",
            "--format=csv,noheader",
        ],
    }
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {
        str(file.relative_to(source_root)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted((source_root / "nanovllm_pp").rglob("*.py"))
        if "docs" not in file.relative_to(source_root).parts
    }
    observations = {}
    for name, command in commands.items():
        try:
            observations[name] = subprocess.check_output(
                command, text=True, stderr=subprocess.STDOUT, timeout=10,
                cwd=source_root,
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            observations[name] = type(exc).__name__
    module_spec = getattr(sys.modules["__main__"], "__spec__", None)
    invocation = (
        ["python", "-m", module_spec.name, *sys.argv[1:]]
        if module_spec
        else ["python", *sys.argv]
    )
    # Preserve the executable/argument structure without publishing local
    # interpreter or checkpoint directories. Hashes identify actual contents.
    if path.is_dir():
        invocation = [
            f"./{path.name}" if argument == str(model_path) else argument
            for argument in invocation
        ]
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(invocation),
        "command_paths_normalized": True,
        "python": sys.version,
        "packages": versions,
        "cuda": cuda_device_info(),
        "model_sha256": hashes,
        "source_sha256": source_hashes,
        "clocks_power_locked": False,
        **observations,
    }
