# TessScope v2.4 final status

## Outcome

V2.4 is complete as a negative validation-only experiment. The frozen-checkpoint audit
successfully corrected v2.3's endpoint-only limitation: 90 of 270 saved checkpoints pass
the unchanged soft ceiling, and three exact checkpoints were frozen before hard labels.
However, none passed every expanded hard and matched-comparison gate. The locked BBBC006
test remained sealed.

## What completed

- All 270 checkpoint records were reconstructed without retraining; 269 unique hashes
  were evaluated once on the unchanged 12-well soft pipeline.
- Ninety checkpoints were soft eligible and 57 were globally nondominated.
- The frozen one-per-run rule selected balanced step 14, action-heavy step 12, and
  first-heavy step 9 from three distinct exact trajectories.
- Three matched stopped-stage controls used identical starts, objectives, schedules, and
  selected step budgets. Exact/stopped forward parity was `0.0`.
- All six frozen pupils received the same 45-well hard endpoint, 27-well primary-density
  subset, one predicted correction, 2,000 paired-well bootstrap replicates, and
  leave-one-well-out stability checks.

## Hard results

| Exact checkpoint | First hard PQ | Focus MAE | Corrected hard PQ | Gain vs piecewise |
|---|---:|---:|---:|---:|
| Balanced, step 14 | 0.492192 | 1.267923 µm | **0.552295** | +0.001285 |
| Action-heavy, step 12 | 0.493238 | 1.251949 µm | 0.551142 | +0.000132 |
| First-heavy, step 9 | **0.500103** | 1.294712 µm | 0.551681 | +0.000671 |

Every exact candidate gained more than `+0.01` first-frame PQ over clear with a positive
paired lower bound, reached at least `98.89%` signed direction accuracy, stayed below
`1.30 µm` MAE, reduced residual defocus, and improved PQ after correction.

The blocking result is the frozen v2.2 piecewise comparison. The required corrected gain
is `+0.005` with a positive paired lower bound; the best observed gain is only
`+0.001285`, with 95% interval `[-0.001719, +0.004800]`. Only `55.6%` of hard wells favor
that candidate, below the frozen 60% stability requirement. Balanced step 14 does show a
real `+0.006259` corrected gain over its matched stopped-stage pupil, with interval
`[+0.000670, +0.011547]`, demonstrating that the feedback gradient is causal but not
enough to beat the strong piecewise reference.

## Decision

No derivative-free run or locked test is authorized because no checkpoint cleared every
earlier hard gate. V2.5 is not activated: its approved trigger was a v2.4 audit with zero
soft-eligible checkpoints, while v2.4 found 90. The final negative freeze is recorded in
`configs/v2_4/pretest-block.json`.
