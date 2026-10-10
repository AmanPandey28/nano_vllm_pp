# Contributing

Keep changes small, readable and backed by tests. This runtime values explicit
ownership and correctness boundaries over production feature breadth.

```bash
python -m pip install -e '.[test,server]'
python -m ruff check nanovllm_pp tests examples
python -m pytest -q -o faulthandler_timeout=30
```

No checkpoint downloads are needed for CPU tests. CUDA tests skip without the
device/dependencies; validate paged changes on a GPU as well:

```bash
python -m nanovllm_pp.benchmarks.gpu_smoke
python -m nanovllm_pp.benchmarks.validate_paged_backend ./Qwen3-0.6B
```

For allocator changes, preserve atomic reservations, exact refcounts, physical
validity publication and immutable full-prefix sharing. For scheduling changes,
test budget/fairness, blocked admission and cancellation. For model/kernel
changes, compare outputs with an oracle at stated dtype tolerances. For HTTP
changes, test partial submission failure and stream cleanup.

Write new measurements under `runs/`, not over recorded fixtures. Performance
claims must include workload shapes/arrivals, warmup, repeats, synchronization,
precision, environment, checkpoint hashes, correctness, negative results and
the exact comparison boundary. GiB and GB are not interchangeable. Do not
commit checkpoints, local documents, credentials or generated package files.

See [design](docs/DESIGN.md), [reproduction](docs/REPRODUCING.md), and
[references](docs/REFERENCES.md) before expanding scope.
