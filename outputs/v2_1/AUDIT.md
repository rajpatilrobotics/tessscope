# TessScope v2.1 causal audit

V2.1 is a validation-only continuation. No BBBC006 test image, label, normalization
value, or metric was accessed.

## Near-miss diagnosis

- Recomputed seed, instance-membership, and total differentiable InstanSeg losses for
  all 18 promoted v2 candidates.
- Total task loss versus hard dense off-focus PQ: Pearson `-0.9681`, Spearman `-0.9705`.
- Instance-membership loss versus hard PQ: Pearson `-0.9592`, Spearman `-0.9581`.
- Exact segmentation/focus gradients conflict at 9/11 sampled points; cosine range is
  `-0.7884` to `+0.7514`, and the `0.03` near-miss is near `-0.62`.
- The `0.03` near-miss beats naive superposition by `+0.02724` hard dense PQ but has
  `0.17571 µm` higher focus MAE.
- One correction raises the near-miss hard PQ from `0.36916` to `0.41667`; naive
  superposition rises from `0.34192` to `0.41407`.
- Leave-one-field-out near-miss hard PQ spans `0.32855` to `0.41485`, so the original
  six-dense-field screen is inadequate for promotion.

Conclusion: soft/hard loss mismatch is not the first blocker. Exact constrained
optimization is warranted by the conflicting branch geometry, while validation must be
expanded. B7 tests whether one additional physically interpretable EDOF mode supplies
the missing optical expressivity.

## B7 gate

- Basis: Noll modes 5–11; piston, tilts, and free defocus remain excluded.
- Constraint: unchanged open `2.5`-radian total RMS coefficient ball.
- Sampled Gram maximum diagonal error: `0.001096`.
- Sampled Gram maximum off-diagonal magnitude: `0.000725`.
- Worst measured PSF support fraction across clear, near-miss, mixed, and spherical
  probes at all seven depths: `0.996785`, above the frozen `0.995` minimum.
- Full served JAX optics → SciPy autofocus → PyTorch InstanSeg derivative:
  relative error `0.002369`, cosine `0.999990`; passed.

## Evidence

- `artifacts/runs/v2_1/audit/near-miss.json`
- `artifacts/runs/v2_1/audit/candidate-table.csv`
- `artifacts/runs/v2_1/audit/gradient-geometry.json`
- `artifacts/runs/v2_1/gates/b7-basis-diagnostics.json`
- `artifacts/runs/v2_1/gates/b7-three-tesseract-derivative.json`
- `outputs/v2_1/near-miss-pareto.png`
- `outputs/v2_1/near-miss-breakdown.png`
- `outputs/v2_1/gradient-geometry.png`

The audit selected exact-gradient constrained B7 continuation followed by expanded,
well-grouped validation. That continuation is now complete; the final negative result is
recorded in [STATUS.md](STATUS.md). Test access remains unauthorized.
