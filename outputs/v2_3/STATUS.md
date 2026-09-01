# TessScope v2.3 closed-loop status

## Outcome

V2.3 is complete as a negative validation-only experiment. The implementation forms two
exposures with the same B7 pupil, differentiates through the frozen autofocus stage
action into the second exposure, and passes the full served directional-derivative gate.
However, none of the nine pre-registered optimization endpoints passed the soft
first-frame segmentation gate. The BBBC006 locked test remained sealed.

## What ran

The complete matrix used three objective profiles (balanced, first-heavy, and
action-heavy), three frozen starts (B7 segmentation-only, v2.1 projected joint, and v2.2
piecewise-028), and 30 exact-gradient Adam steps per profile/start. This is 9 completed
runs and 270 optimization steps. Each endpoint was evaluated on the same 12 fixed
validation wells.

All nine endpoints reduced residual-defocus MAE relative to their start. The strongest
residual endpoint reached `0.934414 µm`, down from `1.325543 µm`, but its first-frame
segmentation loss was `1.116030`, above the frozen maximum of `1.098270`.

## Closest endpoint

The closest soft-gate result was `v2_3-exact-first_heavy-piecewise_028`:

| Metric | Start / requirement | Endpoint |
|---|---:|---:|
| First-frame segmentation loss | maximum `1.098270` | `1.100370` |
| Residual-defocus MAE | `1.191849 µm` | `0.999509 µm` |
| Final-frame segmentation loss | — | `1.061714` |

It improved residual MAE by `0.192339 µm`, but missed the segmentation ceiling by
`0.002099`. The threshold was not relaxed after observing the result.

## Decision

No endpoint advanced to stopped-stage optimization, expanded hard validation, a matched
derivative-free control, or the locked test because those stages were conditional on a
soft-eligible exact endpoint. This preserves the pre-registered decision order and avoids
using hard or test labels to rescue a failed training/validation result.

The negative freeze is recorded in `configs/v2_3/pretest-block.json`; the full nine-run
trace is in `artifacts/runs/v2_3/optimization/closed-loop-matrix.json`.
