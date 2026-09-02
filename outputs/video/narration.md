# TessScope timestamped narration

This script matches the caption-led cut. The rendered release video is intentionally
silent; a presenter can read this narration live or record it later without changing the
scientific edit.

## 00:00–00:22 — Problem and claim

TessScope is a closed-loop fluorescence-microscope co-design for Track 05. A pupil that
protects nucleus segmentation often hides depth, while one that exposes signed defocus
can distort the biology. TessScope optimizes the complete two-exposure decision loop.
On frozen validation, exact feedback gradients improve corrected panoptic quality over a
forward-identical stopped-gradient system.

## 00:22–00:54 — Why Tesseract

Three scientific components stay in three native runtimes. JAX and Chromatix simulate
the phase pupil and fluorescence optics. NumPy and SciPy fit signed depth and predict a
bounded stage action. PyTorch and a frozen InstanSeg model score nucleus utility.
Tesseract gives each component a typed service boundary and derivative endpoint.
Tesseract-JAX makes those remote calls part of one differentiable program, so one exact
reverse signal crosses every API boundary.

## 00:54–01:18 — Optical design

The design is a bounded seven-parameter phase pupil. It forms both the first exposure and
the corrected exposure after the predicted stage move. The derivative therefore includes
the direct effect of the pupil on image formation and the indirect effect of the focus
decision on residual depth. The measured full-loop derivative agrees with central
differences: relative error 0.0048 and cosine 0.99997.

## 01:18–01:46 — Cached validation replay

This animation is a cached, deterministic replay of one preregistered validation field
across seven planes from minus six to plus six micrometres. It is not live inference and
it is not the population-level causal result. The same display transform, field, source
geometry, labels, predictions, and phase designs are used throughout the visual package.

## 01:46–02:14 — Closed-loop action

The first exposure feeds the autofocus Tesseract. Its signed depth estimate becomes a
stage action clipped to plus or minus six micrometres. Residual depth then drives a second
optics call, and InstanSeg evaluates the corrected biological frame. In the primary exact
design, hard off-focus PQ rises from 0.4922 before the action to 0.5523 after it. About
74.7 percent of hard frames improve.

## 02:14–02:38 — Population protocol

The claim is not selected from one attractive picture. All frozen designs use the same
45 registration-valid validation wells, with a preregistered 27-well hard-density subset.
Uncertainty comes from two thousand paired bootstrap replicates that resample wells, not
individual frames. Focus mean absolute error is 1.268 micrometres and signed direction
accuracy is 98.89 percent.

## 02:38–03:16 — Causal result and limitation

Exact and stopped systems have identical forward values. Only the gradient through the
autofocus action is removed. Corrected exact minus stopped PQ is plus 0.006259, with a
95 percent interval from plus 0.000670 to plus 0.011547. That supports the causal value
of the feedback gradient. But exact minus the strong piecewise-028 baseline is only plus
0.001285, with an interval from minus 0.001719 to plus 0.004800. The interval crosses
zero, so TessScope does not claim superiority over that baseline.

## 03:16–03:30 — Close

TessScope demonstrates exact gradients across JAX, SciPy, and PyTorch services in a real
closed-loop optical problem. The result is validation-only. PQ is not percent accuracy,
the BBBC006 test remains sealed, and no physical microscope has yet validated the system.
