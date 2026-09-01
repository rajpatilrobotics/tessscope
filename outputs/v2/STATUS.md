# TessScope v2 status: implementation complete through hard validation

TessScope v2 is not a completed positive hackathon result. It is a reproducible,
evidence-backed **negative pre-test result**: the system works end to end, but no pupil
met every frozen segmentation/focus compromise gate. Locked test images and labels
remain untouched, and no test metric was run.

## What is complete

1. Range-extracted and checksum-validated all 5,376 BBBC006 z13–z19 Hoechst images and
   768 official automated z16 reference masks.
2. Froze 288/48/48 well-level train/validation/test splits before metric access.
3. Registered train/validation stacks; accepted 635/672 fields with `0.2 px` p95
   residual. Low-content fields were excluded transparently.
4. Implemented 10×12 Fourier autofocus features, ridge solve, analytic feature VJP,
   and implicit solve VJP in NumPy/SciPy.
5. Served JAX/Chromatix optics, SciPy autofocus, and PyTorch/InstanSeg as three
   differentiable Tesseracts.
6. Passed the full served v2 derivative contract: `0.007269` relative error and
   `0.999970` cosine agreement.
7. Passed clear real-stack focus (`92.88%`, `1.6227 µm`) and classical astigmatic focus
   (`100%`, `0.3236 µm`) validation gates.
8. Passed the held-out digital-twin calibration gate after fitting one physical axial
   offset: correlation `0.95936`, depth scale `0.95`, offset `-2.5 µm`.
9. Ran clear, cubic, astigmatic, segmentation-only, focus-only, exact joint,
   broken-gradient, naive-superposition, and wall-matched SPSA baselines plus multiple
   exact starts and documented Pareto refinements.
10. Demonstrated the operational decision loop: for the first exact joint pupil, one
    stage correction reduced mean absolute defocus from `3.4286` to `1.0145 µm` and
    raised hard off-focus PQ from `0.3552` to `0.4267`.

## Why the locked test did not run

The first exact joint pupil passed most gates:

- Hard dense off-focus PQ: `0.35521` versus clear `0.33660` (`+0.01861`).
- Signed focus: `100%` direction, `1.0291 µm` MAE.
- Pareto-dominated both naive superposition and matched derivative-free SPSA.
- One correction reduced residual defocus and improved hard PQ.

But segmentation-only reached `0.37894`, so exact joint was `0.02373` lower. The frozen
maximum allowed drop was `0.01`.

The closest refinement from the exact-joint basin moved hard PQ to `0.36916`, reducing
the segmentation-only gap to `0.00979`, while retaining `91.67%` direction and
`1.4399 µm` MAE. It still failed the separate Pareto condition because naive
superposition had lower MAE (`1.2642 µm`). Higher focus weights beat superposition but
again exceeded the segmentation-drop limit. No post-hoc threshold was changed.

## Claim boundary

Supported: a BBBC006-calibrated, hardware-ready in-silico three-Tesseract prototype with
exact cross-framework gradients and a working signed stage-action loop.

Not supported: a learned pupil that satisfies every frozen joint-performance gate,
positive locked-test photon claims, physical microscope performance, or clinical use.

## Reproduce the completed checkpoint

```bash
uv sync
uv run ruff check .
uv run pytest
uv run python scripts/v2_check_autofocus_service.py
uv run python scripts/v2_calibrate_clear_twin.py
```

The served checks require the three local Tesseract services described by the v2
scripts. Do not run a test evaluator unless a new, explicitly approved experiment first
defines a new contract; `configs/v2/pretest-block.json` records the current lock.
