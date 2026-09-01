# TessScope v2.1 final status

## Outcome

The approved v2.1 validation-only implementation is complete through its pre-test
decision point. The result is negative: no B7 or B11 pupil passed every unchanged
segmentation, focus, uncertainty, and matched-baseline gate. No BBBC006 test image,
label, normalization value, or metric was accessed.

This is a completed experiment, not a completed positive learned-pupil claim. V1 remains
the project's locked-test prototype result; v2 and v2.1 are preserved negative pre-test
continuations.

## What was built

- A causal near-miss audit covering hard metrics, focus error, loss alignment, and exact
  branch-gradient geometry.
- B7 Noll 5–11 and B11 Noll 5–15 phase pupils under the unchanged open 2.5-radian RMS
  ball, with piston, tilt, and free defocus excluded.
- Exact served derivatives through JAX optics, NumPy/SciPy autofocus, and PyTorch
  InstanSeg. B7 passed at relative error `0.002369`, cosine `0.999990`; B11 passed at
  `0.002532`, cosine `0.999446`.
- Exact SLSQP epsilon constraints, multiple deterministic starts, matched separate and
  naive-superposition baselines, and segmentation-primary projected gradients.
- Expanded hard validation on the maximum 45 of 48 valid validation wells, with all
  three exclusions disclosed and 2,000 paired bootstrap replicates grouped by well.

## Final gate comparison

| Candidate | Hard PQ | Gain vs clear | Drop vs segmentation | Direction | Focus MAE | Result |
|---|---:|---:|---:|---:|---:|---|
| B7 projected 0.5 | 0.49606 | +0.03995 | 0.00642 | 98.89% | 1.21073 µm | Fails naive-superposition focus |
| B7 projected 1.0 | 0.49440 | +0.03829 | 0.00809 | 98.89% | 1.16210 µm | Fails naive-superposition focus |
| B11 projected 2.0 | 0.49157 | +0.03545 | 0.01206 | 100% | 1.06676 µm | Fails segmentation and superposition |
| B11 projected 3.0 | 0.49044 | +0.03433 | 0.01318 | 100% | 1.05502 µm | Fails segmentation and superposition |

B11 naive superposition reached `0.80987 µm` focus MAE, so neither B11 projected
candidate Pareto-dominated it. Because both B11 candidates also exceeded the maximum
`0.01` hard-PQ drop from B11 segmentation-only, the conditional matched B11
derivative-free run could not change promotion eligibility and was not run.

## Evidence and reproduction

The frozen block and evidence hashes are in
`configs/v2_1/pretest-block.json`. Primary machine-readable results are:

- `artifacts/runs/v2_1/validation/expanded-hard-validation.json`
- `artifacts/runs/v2_1/validation/b11-expanded-hard-validation.json`
- `artifacts/runs/v2_1/optimization/b7-projected-gradient-candidates.json`
- `artifacts/runs/v2_1/optimization/b11-projected-gradient-candidates.json`

Run the lightweight repository verification from the project root:

```bash
uv run ruff check .
uv run pytest -q
```

Regenerating model-dependent evidence additionally requires the ignored BBBC006 data,
InstanSeg bundle, and local Tesseract services documented in the root README. The hard
validator exits nonzero when no candidate passes; that exit is the expected scientific
outcome for both frozen B7 and B11 result files.

## Claim boundary

TessScope v2.1 demonstrates a reproducible three-framework co-design and validation
system, plus a well-supported negative result about this frozen basis/optimizer ladder.
It does not demonstrate a promoted learned pupil, physical microscope performance, or a
successful BBBC006 locked-test result.
