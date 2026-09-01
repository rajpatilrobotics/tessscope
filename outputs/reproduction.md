# TessScope reproduction guide

This guide reproduces the implemented TessScope pipeline on macOS. Commands assume the
terminal is open at the project root.

## 1. Local environment

TessScope targets Python 3.12 and uses `uv`. All Python packages stay inside the
project environment.

```bash
uv sync
uv run python -m tessscope
uv run pytest
uv run ruff check .
```

The verified final state has 45 passing tests and a clean Ruff check.

## 2. External assets

Large assets are intentionally excluded from Git. Download only from the official URLs
recorded in:

- `data/manifests/instanseg-v0.1.2.json`
- `data/manifests/bbbc-assets.json`

Verify every archive SHA256 before extracting it. The code expects:

```text
artifacts/external/instanseg/model-v0.1.2/
  instanseg.pt
  test-input.npy
  test-output_instance_segmentation.npy
  ...the remaining release files

data/external/BBBC039/
  images/
  masks/
  metadata/

data/external/BBBC038/stage1_train/
  <image-id>/images/<image>
```

The checked-in manifests contain exact asset sizes, expected image counts, licenses,
citations, and hashes. Do not use a newer model or silently replace an archive when
reproducing the locked result.

## 3. Observer parity and data evidence

```bash
uv run python scripts/run_gate0.py
uv run python scripts/build_data_manifest.py
uv run python scripts/render_decontam_review.py
uv run python scripts/summarize_training_data.py
uv run python scripts/check_psf_support.py
```

Gate 0 must remain pixel-exact. The frozen exclusion file already records the 43
BBBC039/BBBC038 identity matches and the resulting 75/41/41 train/validation/test
counts. Rebuilding the candidate manifest is for verification; do not alter the frozen
exclusions after inspecting test results.

## 4. Start the two differentiable services

Terminal 1 — JAX/Chromatix optics:

```bash
TESSERACT_API_PATH=services/optics/tesseract_api.py \
  uv run tesseract-runtime \
  --output-path artifacts/runtime-runs/optics serve --port 8401
```

Terminal 2 — PyTorch/InstanSeg observer:

```bash
TESSERACT_API_PATH=services/observer/tesseract_api.py \
  uv run tesseract-runtime \
  --output-path artifacts/runtime-runs/observer serve --port 8402
```

Leave both terminals running for Gates 5, 7, and 8.

## 5. Verify the served derivative

```bash
uv run python scripts/check_component_derivatives.py
uv run python scripts/check_served_derivative.py
```

Expected full-chain stable-window result:

- median relative error: approximately `0.0052` (required `< 0.01`)
- cosine agreement: approximately `0.99961` (required `> 0.99`)

## 6. Validation-only choices and matched designs

These commands are computational experiments, not fast smoke tests:

```bash
uv run python scripts/benchmark_evaluation_route.py
uv run python scripts/select_observer_transform.py
uv run python scripts/select_cubic_strength.py
uv run python scripts/calibrate_optimizer.py
uv run python scripts/run_matched_designs.py
```

The exact-task, image-fidelity, and calibrated-surrogate designs each use 120 Adam steps
from the same zero start, schedule, phase family, learning rate, and step budget.

## 7. Locked test evaluation

`scripts/evaluate_locked_test.py` verifies every pre-test hash before reading the test
split. On the verified MacBook Air M2 it takes about 35 minutes and writes 4,715 metric
rows. Running it again is unnecessary unless you deliberately want a full independent
reproduction.

```bash
uv run python scripts/evaluate_locked_test.py
```

The machine-readable local evidence is written under `artifacts/runs/gate9/`:

- `raw-metrics.csv`
- `raw-metrics.json`
- `test-report.json`

Generated experiment evidence is ignored by Git; the checked-in readable report in
`outputs/TessScope-results.md` contains the audited result.

## 8. Regenerate figures

This command reads the completed gate artifacts and evaluates only the first frozen
test source for the qualitative panel. It does not change the locked report.

```bash
uv run python scripts/generate_outputs.py
```

It writes the five PNG figures in `outputs/`.
