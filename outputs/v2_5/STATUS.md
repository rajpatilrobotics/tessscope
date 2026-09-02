# TessScope v2.5 final status

## Outcome

V2.5 is complete as a negative training/validation experiment. The exact constrained
B11 method and B7 continuity control were fully implemented and run under the frozen
contract, but neither basis produced a candidate satisfying the matched training
constraints and unchanged 12-well soft-validation ceiling. Zero candidates were
promoted, and the locked BBBC006 test remained sealed.

This completes the approved v2.5 research scope. It does not mean that TessScope has a
new positive learned pupil, physical-microscope validation, or a polished deployed
hackathon application.

## What completed

- Froze the v1–v2.4 sources, B7-to-B11 lifts, five B11 starts, two B7 starts, three
  residual bounds, optimizer budgets, gates, and test policy before optimization.
- Implemented the differentiable-depth B11 optics service and exact constrained
  augmented-Lagrangian/SLSQP utilities.
- Passed the B11 closed-loop derivative gate: `0.009022` relative error, `0.999887`
  cosine agreement, `43.06%` stage-path gradient fraction, and exact forward parity.
- Completed 15 B11 primary endpoints and 270 aggregate exact-gradient steps.
- Completed six B7 primary endpoints and 108 aggregate exact-gradient steps.
- Activated the preregistered fallback for both bases and completed six capped SLSQP
  runs, 48 total iterations, and 105 exact vector evaluations.
- Applied the frozen 12-well soft gate and selected zero B11 candidates.

## Gate result

| Basis | Primary endpoints | Primary steps | Fallback runs | Soft eligible | Selected |
|---|---:|---:|---:|---:|---:|
| B11 | 15 | 270 | 3 | 0 | 0 |
| B7 continuity | 6 | 108 | 3 | 0 | 0 |

The optimizer found individual improvements, including B11 validation residual MAE as
low as `0.7683 µm` and B7 as low as `0.9397 µm`. Those points did not simultaneously
preserve the registered first-frame segmentation requirement. The best observed B11
first-frame loss was `1.106055` versus the `1.098196` maximum; the best B7 value was
`1.103634` versus `1.098270`.

## Decision

The contract required a selected B11 candidate before the matched B11 piecewise,
stopped-stage, hard-validation, derivative-free, controller-gain, and locked-test
branches. With zero selections, those branches were correctly not run. This is a clean
negative result, not an unfinished numerical matrix.

The machine-readable decision and evidence hashes are in
`configs/v2_5/pretest-block.json`. The B11 and B7 result artifacts are in
`artifacts/runs/v2_5/optimization/`.

## Reproduce the local checks

From the repository root, with the project environment already installed:

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
git diff --check
```

The large model and dataset inputs remain local and are not bundled into Git.
