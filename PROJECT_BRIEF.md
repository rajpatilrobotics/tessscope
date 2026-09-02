# TessScope project brief

## Hackathon release candidate

TessScope is prepared for **Tesseract Hackathon 2026, Track 05 — Differentiable graphics
& rendering**. The protected result is validation-scoped: exact gradients across
JAX/Chromatix, NumPy/SciPy autofocus, and PyTorch/InstanSeg causally improve closed-loop
correction over a forward-identical stopped-gradient system by `+0.006259` PQ, with a
95% well-bootstrap interval of `[+0.000670, +0.011547]` on 27 hard-density validation
wells.

The stronger exact-versus-piecewise comparison is not supported, the BBBC006 test stays
sealed, and no physical microscope claim is made. The judge path is the dependency-free
cached replay launched by `python3 scripts/serve_demo.py`; the technical brief, video,
full reproduction guide, and research history are linked from the root README.

Local release status: **GO**. All approved package, clean-export, test, traceability,
licensing, PDF, and video gates pass. No remote, publication, upload, deployment, or
hackathon submission has been performed.

## V2.6 diagnosis-first improvement

V2.6 preserves v1–v2.5 and targets the principal remaining scientific blocker: a
statistically credible corrected-PQ advantage over the strongest fair piecewise system,
not merely a lower focus error. It first measures the frozen balanced pupil's oracle
correction ceiling, PQ-versus-residual-depth curve, controller error structure, and
fixed-total-photon exposure headroom. Only the intervention supported by that frozen
diagnostic may run.

The primary route is an interpretable bounded gain/bias/monotone controller calibration,
optionally with exposure allocation, jointly optimized with the B7 pupil only if the
oracle demonstrates sufficient headroom. Piecewise-028 receives the same controller and
exposure budget. If that route cannot reach the minimum `+0.005` corrected-PQ effect with
grouped confidence and stability, v2.6 freezes it and considers sequential two-mask
acquisition only after physical feasibility research. The locked test remains sealed
until all applicable prerequisites pass.

Diagnosis result: the single-pupil controller/exposure route is rejected. Its exact
zero-residual oracle reaches only `0.554982` corrected hard PQ, and an equally budgeted
monotone controller grid gains only `+0.001007` over piecewise. A fixed-total-photon
training confirmation audit favors piecewise by `0.006473`. Primary literature and
manufacturer specifications nevertheless support a physically qualified sequential
two-mask route, so v2.6 continues with separately frozen sensing and capture pupils under
the same 400-photon budget and explicit SLM switching/efficiency limitations.

Final v2.6 status: the two-mask full-loop derivative passed in joint, sensing-only, and
capture-only directions. All 76 matched piecewise pairs were evaluated, but none met the
frozen residual gate; the best residual MAE was `1.526 µm`. Both exact starts then
completed four eight-step feasibility restorations and all 72 primary steps. Zero
restorations were feasible, every development endpoint had residual MAE near `2.0 µm`,
and zero endpoints qualified for confirmation. V2.6 is therefore a complete negative
training-only experiment. No new validation/hard/test data was accessed.

## V2.5 constrained B11 experiment

V2.5 is a newly and explicitly approved experiment motivated by the frozen v2.4 hard
near miss. It preserves v1–v2.4 and asks whether a transparent exact constrained optimizer
on the manufacturable B11 Noll 5–15 pupil can improve the corrected frame while directly
enforcing the unchanged first-frame segmentation ceiling and a registered residual-focus
bound. B7 remains a continuity control; B11 is the primary basis.

The experiment uses a fixed three-level residual epsilon ladder, five frozen B11 starts,
two B7 continuity starts, a preregistered exact SLSQP alternate, a strong B11 piecewise
family, and matched stopped-stage/derivative-free controls. Soft selection may freeze at
most three B11 candidates from the same 12 validation wells. Every v2.4 hard threshold is
unchanged. The locked BBBC006 test remains sealed unless all validation, causal-gradient,
piecewise, stability, and derivative-free gates pass and a pretest commit is verified.

Final v2.5 status: the B11 derivative gate passed, then all 15 B11 and six B7 primary
constrained stages completed 378 exact aggregate steps. Neither basis produced a
training-and-soft-eligible endpoint. Six preregistered SLSQP fallback runs also failed
their matched constraints, so zero B11 candidates were frozen. The conditional
piecewise, stopped-stage, hard-validation, derivative-free, gain, and locked-test work
was correctly skipped. V2.5 is a complete negative training/validation experiment, not a
positive learned-pupil result or a finished judge-facing application.

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

Final v2.4 status: early stopping found 90 soft-eligible checkpoints and froze three
exact candidates. All three improved first-frame hard PQ over clear with positive paired
confidence, achieved at least `98.89%` direction accuracy and below `1.30 µm` MAE, and
improved the corrected frame. The best corrected PQ was `0.552295`, but its gain over
v2.2 piecewise-028 was only `+0.001285` with interval
`[-0.001719, +0.004800]`, below the required `+0.005` and positive lower bound. V2.4 is
therefore a complete negative validation result; derivative-free and locked-test work
were not authorized.

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
