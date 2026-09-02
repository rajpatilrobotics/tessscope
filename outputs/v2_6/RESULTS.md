# TessScope v2.6 result

## Decision

V2.6 is complete as a **negative training-only experiment**. It did not produce a pupil,
controller, exposure policy, or sequential-mask pair that is eligible for new hard or
locked-test evaluation. The locked BBBC006 test remains sealed.

## What was completed

1. The frozen single-pupil oracle reached corrected PQ `0.554982`, below the required
   `0.556010` minimum implied by the existing piecewise baseline.
2. A matched controller grid gained only `+0.001007` over piecewise and had a confidence
   interval crossing zero.
3. The fixed 400-photon exposure audit favored piecewise by `0.006473` on the frozen
   training confirmation partition.
4. Primary literature and official device specifications supported a qualified
   sequential two-SLM-mask simulation. Hardware switching, efficiency, bleaching,
   motion, and calibration remain unvalidated.
5. The sequential graph passed the exact derivative gates:
   - joint relative error `0.003168`, cosine `0.999985`;
   - sensing-only relative error `0.005255`, cosine `0.999978`;
   - capture-only relative error `0.001991`, cosine `0.999997`;
   - exact/stopped forward difference `0`; stage-path gradient fraction `1.0`.
6. All 76 matched two-mask piecewise pairs were screened. All were physically valid and
   42 preserved the first-frame limit, but zero passed the residual gate. Best residual
   MAE was `1.525840 µm`, above the `1.0 µm` requirement.
7. Both exact starts completed four eight-step feasibility restorations and the complete
   72-step primary matrix. No restoration was feasible and no endpoint qualified for
   confirmation.

## Closest exact endpoints

| Endpoint | Development first loss | Development final loss | Residual MAE |
|---|---:|---:|---:|
| Piecewise start, 0.055 stage | 1.124608 | **1.098068** | 2.014347 µm |
| Balanced start, 0.055 stage | **1.120328** | 1.100178 | **1.997605 µm** |

The development first-frame limit was `1.121831`. The piecewise endpoint improved final
loss but missed both first-frame and residual gates. The balanced endpoint preserved the
first-frame limit but remained far outside the `0.055` normalized residual and `1.0 µm`
MAE requirements.

## Branches correctly not activated

- No confirmation candidate selection.
- No new validation or hard-label evaluation.
- No matched stopped-stage or derivative-free control.
- No locked-test access or rerun.
- No deployment, publication, submission, or push.

Repository verification passes: Ruff reports no issues and all 177 tests pass. The only
test output is four existing upstream PyTorch/InstanSeg deprecation or sparse warnings.

## Evidence

- `artifacts/runs/v2_6/diagnostics/oracle-controller-audit.json`
- `artifacts/runs/v2_6/diagnostics/exposure-audit.json`
- `artifacts/runs/v2_6/gates/two-mask-derivative.json`
- `artifacts/runs/v2_6/two-mask/matched-baseline-screen.json`
- `artifacts/runs/v2_6/two-mask/exact-optimization.json`
- `configs/v2_6/pretest-block.json`
