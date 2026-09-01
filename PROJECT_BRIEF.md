# TessScope project brief

## V2.4 frozen-checkpoint experiment

V2.4 is a separately named, approved validation-only audit of the 270 intermediate phase
vectors already saved by the frozen v2.3 exact-gradient runs. It does not retrain v2.3,
change its endpoint conclusion, or relax the first-frame segmentation ceiling. Soft
validation may select at most three nondominated checkpoints under a pre-registered
one-per-run diversity rule; hard labels remain a later promotion gate.

If no frozen checkpoint satisfies both the unchanged segmentation ceiling and improved
residual defocus, v2.4 freezes a negative result and automatically activates separately
named v2.5 constrained closed-loop optimization. The locked BBBC006 test remains sealed
until every applicable validation and matched-comparison gate passes.

## V2.2 matched-frontier experiment

V2.2 is an approved, separately named validation-only experiment. It freezes v2.1's
strongest B7 joint pupil and tests it against a complete physically projected family of
piecewise combinations of the matched segmentation-only and focus-only B7 pupils. It
changes no v2.1 result or threshold; it replaces only v2.2's scientifically inadequate
comparison to one extreme sum with a predeclared matched Pareto-envelope test.

The strongest demo question is now precise: at the same first-frame segmentation
quality, does differentiating the joint task produce a more useful stage decision than
any piecewise combination of separately optimized optics? The BBBC006 locked test remains
sealed unless this validation comparison and all inherited gates pass before a local
pre-test commit.

Final v2.2 status: no. Piecewise-028 reaches hard PQ `0.502543` and focus MAE
`1.20718 µm`, slightly improving both over the frozen joint's `0.496064` and
`1.21073 µm`. The joint adds zero normalized hypervolume, so v2.2 is preserved as a
negative validation result and the approved v2.3 closed-loop experiment is active.

## V2.3 closed-loop differentiable microscope

V2.3 makes the feedback action part of the optimized computation: the same B7 pupil
forms a first exposure, the SciPy autofocus Tesseract predicts a bounded stage move, the
move changes residual depth, the same pupil forms a second exposure, and InstanSeg scores
the corrected biological frame. Exact gradients must cross the predicted action into the
second Chromatix call. A forward-identical stopped-stage-gradient run tests whether that
feedback path matters rather than merely decorating the architecture.

Final v2.3 status: the closed loop and its exact served derivative are implemented and
validated, and the full 3-profile × 3-start matrix completed 270 Adam steps. All nine
endpoints reduced residual-defocus MAE, but all exceeded the frozen first-frame
segmentation-loss ceiling. The closest endpoint missed by `0.002099`, so no endpoint
advanced to hard validation or the locked test. V2.3 is preserved as a complete negative
validation experiment, not described as a positive learned-pupil result.

## V2.1 continuation

V2.1 is a separately named, validation-only continuation motivated by the frozen v2
near-miss. It preserves v1 and v2 as scientific history, keeps the BBBC006 test images
and labels sealed, and asks whether a slightly richer manufacturable pupil plus an
exact-gradient constrained optimizer can enter the already frozen joint-performance
region. The success thresholds below are unchanged.

Final status: the approved B7→B11 ladder is complete as a negative pre-test result. Both
bases produced useful Pareto candidates, but no single pupil passed every frozen gate.
The locked test was therefore not accessed, and no learned-pupil headline is claimed.

## One-line idea

TessScope v2 co-designs one bounded phase-only fluorescence-microscope pupil so a
single photon-matched exposure both preserves nucleus segmentation and tells the stage
which direction and distance to move toward focus.

## Frozen scientific question

Can a manufacturable phase-only pupil make one photon-matched fluorescence exposure
simultaneously useful for nucleus segmentation and informative enough to tell the
microscope stage which direction and distance to move?

The claim boundary is a **BBBC006-calibrated, hardware-ready in-silico optical design**.
The learned mask is not fabricated, installed, or physically validated.

## User and demo audience

The primary audience is a Tesseract Hackathon 2026 judge or microscopy researcher. A
judge should understand the real focus-versus-segmentation conflict, see two observer
cotangents cross three framework/process boundaries, inspect a learned phase pupil, and
watch one predicted stage correction improve a second frame.

## Non-piecewise scientific conflict

- Segmentation-only extended-depth-of-field optics prefer depth-invariant images.
- Signed single-shot autofocus requires depth-dependent, sign-identifiable image cues.
- Making all depths look alike destroys the stage decision.
- Encoding strong depth cues can distort nuclei and damage instance segmentation.

The joint exact gradient must therefore search a Pareto frontier. Independently
optimized focus and segmentation pupils, naively superposed under the same constraints,
are an explicit baseline rather than an assumed solution.

## Three-Tesseract architecture

1. **JAX/Chromatix optics:** calibrated incoherent fluorescence image formation and a
   low-dimensional phase pupil with coefficient, RMS, throughput, and quantization
   constraints. It receives the sum of both observer cotangents.
2. **NumPy/SciPy autofocus:** stabilized content-normalized Fourier radial/angular
   features and a regularized signed-z ridge solver. Its FFT, aggregation, normalization,
   and ridge solve use a hand-derived analytic/implicit VJP, not hidden JAX or PyTorch
   autograd.
3. **PyTorch/InstanSeg observer:** the frozen `single_channel_nuclei` v0.1.2 raw-head
   task loss and exact input VJP. Official hard instance postprocessing remains outside
   optimization and is used for final metrics.

One composed objective must return one exact pupil gradient through the served chain.
Tesseract is load-bearing because the tested scientific components remain in their
native differentiation ecosystems rather than being reimplemented with parity risk.

## Primary dataset and evaluation unit

- BBBC006 U2OS high-content fluorescence z-stacks, Hoechst channel only.
- Seven planes z13–z19 around laser-reference focus z16: `−6` through `+6 µm` in
  `2 µm` steps.
- All planes and sites from one well remain in one deterministic split.
- Calibration and tuning use training/validation wells only.
- One final test evaluation is paired and bootstrapped by well.
- BBBC006 reference masks are automated CellProfiler-derived masks, not manual ground
  truth.

BBBC039 remains frozen v1 continuity and exploratory evidence, not a newly untouched v2
test set.

## Strongest demo moment

For one frozen held-out BBBC006 field, show the first photon-matched exposure, hard
nucleus instances, predicted signed defocus and stage arrow, the nearest available
post-correction plane, and before/after residual defocus and PQ. A Pareto plot compares
clear, classical, separate, superposed, derivative-free, surrogate, and exact joint
designs.

## Required baselines

1. Clear pupil.
2. Classical cubic EDOF pupil.
3. Classical astigmatic focus pupil.
4. Segmentation-only exact pupil.
5. Focus-only exact pupil.
6. Naively superposed separate pupils under matched constraints.
7. Full joint exact TessScope pupil.
8. Forward-identical broken/surrogate-gradient ablation.
9. Matched random or CMA-ES derivative-free search.
10. Multiple starts and a small objective-weight Pareto sweep when affordable.

## Frozen success gates

- Every component VJP and the full three-Tesseract served derivative pass finite
  differences; full-chain relative error `< 1e-2`, cosine `> 0.99`.
- Held-out nonzero-depth signed-direction accuracy `≥ 90%` and focus MAE `≤ 2 µm`.
- Joint mean off-focus PQ is at least clear `+0.01` with a well-grouped confidence
  interval.
- Joint PQ is no worse than segmentation-only by more than `0.01` while autofocus is
  substantially better.
- Joint Pareto-dominates naive superposition and matched derivative-free search.
- One predicted stage correction reduces absolute residual defocus and improves PQ.
- Photon evidence is positive at both 50 and 200 photons under the frozen calibrated
  definitions.
- A clear-pupil digital-twin calibration gate passes held-out real BBBC006 stacks,
  targeting depth-curve correlation `≥ 0.90` before opening final test results.

Failed gates remain negative results; thresholds are never changed after test access.

## V1 preservation

V1 is complete and immutable: 41 decontaminated BBBC039 test sources, 4,715 observer
images, exact-task off-focus PQ `0.6098` versus clear `0.5873`, and exact VJP beating a
forward-identical surrogate by `+0.0201`. Its count and photon gates failed and remain
disclosed. Original v1 files and evidence are hashed in `experiments/v1/freeze.json`.
V2 uses separately named modules, configs, scripts, and artifact paths.

## Constraints

- MacBook Air M2 is the default compute platform; external GPU is justified only by a
  measured runtime or memory gate.
- Stream/extract only required BBBC006 Hoechst planes and keep disk usage bounded.
- No hidden per-image normalization, test tuning, fabricated hardware claim, clinical
  claim, destructive operation, global install, cloud/billing action, push, or
  publication without explicit approval. Verified local milestone commits are approved.
- Final eligibility, submission wording, and polish follow scientific completion.
