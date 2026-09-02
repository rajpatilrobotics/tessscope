# Full TessScope reproduction

This is the scientific path behind the instant cached judge replay. It is intentionally
separate because it needs external data, a frozen model, three local services, and much
more computation. None of these commands accesses the sealed BBBC006 test split.

Commands assume a terminal at the repository root on macOS Apple silicon.

## 1. Requirements and resource envelope

- Python `3.12` (the project requires `>=3.12,<3.13`).
- `uv` with the committed `uv.lock`; the verified local version is `0.11.28`.
- Network access only for the official model and dataset acquisition step.
- About 2.2 GB for the training+validation BBBC006 W1 materialization; the complete
  seven-plane W1 subset is approximately 2.5 GB but is not needed and is not committed.
- About 15 MB for the InstanSeg `single_channel_nuclei` v0.1.2 bundle.
- Three localhost ports: `8407`, `8403`, and `8402`.
- No Docker, cloud account, API key, or GPU is required. Apple MPS is used when available.

Create the exact local environment:

```bash
uv sync --frozen
uv run python --version
```

The second command must report Python 3.12.x. Package versions and the Chromatix commit
are locked in [`uv.lock`](../uv.lock) and [`pyproject.toml`](../pyproject.toml).

## 2. Verify the repository before downloading anything

```bash
python3 scripts/serve_demo.py --check
uv run python scripts/audit_release_candidate.py --history-secrets
uv run ruff check .
uv run pytest -q
```

The release audit must report `status=GO`, zero blockers, and `test_accessed=false`.

## 3. Fetch official external assets

The model helper downloads the exact upstream release, checks the archive SHA-256, rejects
unsafe ZIP paths, extracts only the expected files, and verifies every file hash:

```bash
uv run python scripts/fetch_instanseg.py
```

Fetch the BBBC006 automated reference-label archive and verify its published local
manifest hash:

```bash
uv run python scripts/fetch_bbbc006_labels.py
```

Materialize only training and validation Hoechst planes. The range-based downloader reads
the central directory of each official ZIP, fetches W1 members only, checks ZIP CRC32,
image shape and type, and records SHA-256 values. Passing the split list is important:

```bash
uv run python scripts/v2_fetch_bbbc006.py --splits training validation --workers 6
```

Six workers is a conservative setting for a MacBook Air. The tracked complete member
inventory in [`data/manifests/v2/bbbc006-w1-members.json`](../data/manifests/v2/bbbc006-w1-members.json)
is provenance evidence; the downloaded TIFFs and model files remain Git-ignored.

The code expects:

```text
artifacts/external/instanseg/model-v0.1.2/instanseg.pt
data/external/BBBC006/v1/images/z_13/ ... z_19/
data/external/BBBC006/v1/labels/
```

Do not add `test` to the fetch command and do not run `scripts/evaluate_locked_test.py`.
That older script belongs to the separate v1 BBBC039 history, not the v2 public claim.

## 4. Check model parity and prepared-data assumptions

```bash
uv run python scripts/run_gate0.py
```

Gate 0 must report pixel-exact parity with the upstream InstanSeg release example. The
training-only normalization, well split, registration, and calibration artifacts are
already frozen and tracked under `data/manifests/v2/` and `artifacts/runs/v2/`.

## 5. Start the three Tesseracts

Open three terminals at the repository root and keep them running.

Terminal 1 — JAX/Chromatix closed-loop optics:

```bash
TESSERACT_API_PATH=services/v2_3/optics/tesseract_api.py \
  uv run tesseract-runtime \
  --output-path artifacts/runtime-runs/reproduction/optics serve --port 8407
```

Terminal 2 — NumPy/SciPy autofocus:

```bash
TESSERACT_API_PATH=services/v2/autofocus/tesseract_api.py \
  uv run tesseract-runtime \
  --output-path artifacts/runtime-runs/reproduction/autofocus serve --port 8403
```

Terminal 3 — PyTorch/InstanSeg observer:

```bash
TESSERACT_API_PATH=services/v2_1/observer/tesseract_api.py \
  uv run tesseract-runtime \
  --output-path artifacts/runtime-runs/reproduction/observer serve --port 8402
```

Each process should reach a healthy serving state. The first request may be slower while
JAX and PyTorch compile or warm their models.

## 6. Reproduce the end-to-end derivative gate

Write the rerun to an isolated, ignored location so the frozen evidence remains unchanged:

```bash
uv run python scripts/v2_3_check_closed_loop_derivative.py \
  --output artifacts/runtime-runs/reproduction/closed-loop-derivative.json
```

Required checks:

- median relative error `< 0.01`;
- cosine agreement `> 0.99`;
- stage-path gradient fraction `>= 0.01`;
- exact/stopped forward difference `<= 1e-6`;
- `test_accessed=false`.

The frozen run measured `0.004802` relative error, `0.999967` cosine, and `0.0` forward
difference.

## 7. Optional: rerun the v2.3 optimization matrix

This is not needed to inspect or verify the frozen v2.4 claim, but it independently reruns
the nine exact trajectories into isolated resumable checkpoints:

```bash
uv run python scripts/v2_3_optimize_closed_loop.py \
  --output artifacts/runtime-runs/reproduction/closed-loop-matrix.json \
  --checkpoint-root artifacts/runtime-runs/reproduction/v2_3-checkpoints
```

The historical nine exact runs contain 270 Adam steps and took about 16 aggregate minutes
on the project MacBook Air M2 after warm-up. Runtime varies with compilation and memory
pressure. The script resumes its isolated per-run checkpoints if interrupted.

## 8. Reproduce the primary v2.4 validation result

This evaluates the already frozen exact and stopped pupils on the same 45 permitted
validation wells, reconstructs the 27-well density subset, and repeats the deterministic
2,000-replicate paired-well bootstrap:

```bash
uv run python scripts/v2_4_validate_expanded_hard.py \
  --output artifacts/runtime-runs/reproduction/expanded-hard-validation.json \
  --checkpoint-root artifacts/runtime-runs/reproduction/v2_4-hard-checkpoints
```

This is the longest reproduction stage. It performs repeated InstanSeg hard inference;
allow hours rather than minutes, keep the MacBook on power, and close memory-heavy apps.
It is resumable at the per-design checkpoint level.

Compare the rerun's public metrics with the frozen source:

```bash
uv run python scripts/verify_v2_4_reproduction.py \
  artifacts/runtime-runs/reproduction/expanded-hard-validation.json
```

The comparison uses a default absolute tolerance of `1e-4` for cross-device numerical
variation and fails if the report indicates test access.

## 9. Rebuild the cached judge evidence

These deterministic commands use only permitted frozen evidence:

```bash
uv run python scripts/build_demo_traceability.py
uv run python scripts/select_demo_evidence.py
uv run python scripts/build_demo_evidence.py
uv run python scripts/generate_demo_figures.py
uv run python scripts/build_demo_site.py
uv run python scripts/build_demo_traceability.py
python3 scripts/serve_demo.py --check
```

The first traceability call verifies the source contract; the final call closes the
manifest chain after all generated files exist. A clean Git diff after rebuilding proves
byte-stability of the tracked replay, figures, and site.

## Troubleshooting

### A port is already in use

Stop the process using that local port, or pass matching custom service URLs to the
derivative and optimization commands. The canonical ports are retained above so the
commands work without extra flags.

### A partial download exists

The fetch helpers deliberately refuse to overwrite `.partial` files. Inspect the named
file and move it aside manually before retrying; this prevents a corrupt or unexpectedly
replaced upstream asset from being accepted silently.

### The process is killed or memory pressure is high

Use six or fewer data-download workers, run only one scientific driver at a time, and
keep the three services warm. The optimization and hard-validation drivers use isolated
resumable checkpoints.

### A result differs beyond tolerance

Confirm Python 3.12, `uv sync --frozen`, exact model hashes, the committed well split,
and the three correct service entry points. Do not tune a threshold or substitute a
newer model. Preserve the differing report and investigate it as a reproduction result.

## Claim boundary

Successful reproduction supports only the frozen validation claim. It does not convert
PQ into percent accuracy, establish superiority over piecewise-028, validate physical
hardware, or authorize opening the sealed BBBC006 test.
