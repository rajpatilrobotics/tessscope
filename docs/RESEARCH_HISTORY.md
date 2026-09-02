# TessScope research history

This document preserves the experimental arc behind the concise judge-facing README.
Negative results remain visible because they define the claim boundary and show that no
gate was relaxed after validation evidence was seen.

## Current public result: v2.4

V2.4 audited all 270 saved checkpoints from the frozen v2.3 exact-gradient matrix without
retraining. Ninety checkpoints passed the unchanged soft ceiling; three were frozen by a
preregistered one-per-run rule before hard-label evaluation. Each received a matched
forward-identical stopped-stage control and the same 45-well hard validation protocol.

The balanced step-14 checkpoint is the strongest causal result:

| Measure | Value |
|---|---:|
| First hard off-focus PQ | `0.492192` |
| Corrected hard off-focus PQ | `0.552295` |
| Focus MAE | `1.267923 µm` |
| Signed-direction accuracy | `98.89%` |
| Corrected exact vs. stopped | `+0.006259`, CI `[+0.000670, +0.011547]` |
| Corrected exact vs. piecewise-028 | `+0.001285`, CI `[−0.001719, +0.004800]` |

The stopped comparison supports the causal value of the feedback gradient. The
piecewise comparison does not support superiority over the strongest matched baseline.
Only 55.6% of hard wells favored the exact design over piecewise, below the frozen 60%
stability rule. The first-frame segmentation tolerance also missed by `0.000295` PQ.

Evidence: [v2.4 status](../outputs/v2_4/STATUS.md),
[checkpoint audit](../outputs/v2_4/CHECKPOINT_AUDIT.md), and
[`expanded-hard-validation.json`](../artifacts/runs/v2_4/validation/expanded-hard-validation.json).

## v2.6: diagnosis-first and sequential masks

V2.6 first tested whether controller calibration or exposure allocation could provide
enough headroom. The single-pupil zero-residual oracle reached corrected PQ `0.554982`,
and the matched monotone controller gained only `+0.001007` over piecewise. A frozen
400-photon training confirmation audit favored piecewise by `0.006473`.

A separately preregistered sequential sensing/capture-mask route then passed joint,
sensing-only, and capture-only derivative gates. All 76 matched two-mask piecewise pairs
and the complete 72-step exact matrix were evaluated. Zero candidates met the frozen
residual and first-frame requirements, so no confirmation, hard validation, stopped
control, derivative-free control, or locked-test branch was activated.

Evidence: [v2.6 results](../outputs/v2_6/RESULTS.md) and
[SLM feasibility review](../outputs/v2_6/SLM_FEASIBILITY.md).

## v2.5: constrained B11 continuation

V2.5 asked whether a larger Noll 5–15 pupil and exact constrained optimization could
resolve the v2.4 near miss. The full B11 served derivative passed (`0.009022` relative
error, `0.999887` cosine). Fifteen B11 and six B7 primary endpoints completed 378 exact
aggregate steps, followed by six preregistered capped SLSQP fallbacks. Zero endpoints
satisfied both the matched training constraints and the unchanged 12-well soft ceiling.

This is a complete negative training/validation experiment, not an abandoned run.
Evidence: [v2.5 status](../outputs/v2_5/STATUS.md) and
[v2.5 audit](../outputs/v2_5/AUDIT.md).

## v2.3: the differentiable feedback loop

V2.3 introduced the loop used in the public claim: first exposure → SciPy focus action →
residual depth → second exposure → InstanSeg loss. The derivative crossed all three
Tesseracts with `0.004802` median relative error, `0.999967` cosine, and `0.0` exact/stopped
forward difference.

Nine preregistered exact runs (three objective profiles × three starts) completed 270
Adam steps. Every endpoint reduced residual defocus, but every endpoint exceeded the
frozen first-frame segmentation-loss ceiling. V2.4 later audited the already-saved
intermediate checkpoints without changing v2.3's endpoint conclusion.

Evidence: [v2.3 status](../outputs/v2_3/STATUS.md) and
[`closed-loop-derivative.json`](../artifacts/runs/v2_3/gates/closed-loop-derivative.json).

## v2.2: matched piecewise frontier

V2.2 replaced a weak naive-superposition comparison with 38 frozen, physically projected
piecewise points made from matched B7 segmentation and focus pupils. Piecewise-028
slightly improved both hard PQ and focus MAE over the frozen joint candidate. The joint
added zero normalized hypervolume, so no locked-test access was allowed.

Evidence: [v2.2 status](../outputs/v2_2/STATUS.md).

## v2 and v2.1: BBBC006 three-Tesseract foundation

V2 established the BBBC006 well split, range-based Hoechst acquisition, z-stack
registration, training-only normalization, clear-twin calibration, B7 phase optics,
analytic/implicit SciPy autofocus VJP, frozen InstanSeg observer, matched baselines, and
the first full three-Tesseract derivative gate.

V2.1 expanded the hard validation protocol and the phase basis from B7 to B11. Useful
Pareto points emerged, but no single pupil simultaneously passed segmentation, focus,
and matched-frontier gates. The locked BBBC006 test therefore remained sealed.

Evidence: [v2 status](../outputs/v2/STATUS.md),
[v2.1 status](../outputs/v2_1/STATUS.md), and
[v2.1 audit](../outputs/v2_1/AUDIT.md).

## v1: two-Tesseract continuity study

V1 used BBBC039 with a JAX/Chromatix optics Tesseract and a PyTorch/InstanSeg observer
Tesseract. Its separately locked evaluation found exact-task off-focus PQ `0.6098` versus
clear `0.5873`, but did not meet its preregistered `+0.05` headline threshold and did not
improve every secondary endpoint. The full v1 tree was frozen before BBBC006 v2 work.

Evidence: [v1 report](../outputs/TessScope-results.md) and
[`experiments/v1/freeze.json`](../experiments/v1/freeze.json).

## Unchanged boundary

- V2 evidence is validation-only.
- The BBBC006 test images, labels, and metrics remain sealed.
- Automated CellProfiler reference labels are not manual ground truth.
- No physical microscope or fabricated phase mask has validated the simulation.
- The judge replay is cached evidence, not live inference.
- Failed and non-activated branches are reported as such; they are not counted as wins.

The chronological preregistrations and decisions remain in
[`decision-log.md`](../decision-log.md) and [`plan.md`](../plan.md).
