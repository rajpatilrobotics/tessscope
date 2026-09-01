# TessScope v2.1 continuation plan

## 1. Goal

Continue the frozen BBBC006 three-Tesseract prototype with a separately named v2.1
experiment. Find the smallest manufacturable pupil and exact-gradient constrained
optimization method that preserves segmentation within the frozen tolerance while
beating naive superposition on signed focus.

The full v2 handoff is approved. Work proceeds autonomously unless a consequential
scientific choice cannot be resolved on training/validation data, or a destructive,
legal, credential, billing, publication, conflicting-user-work, or external blocker
requires the user.

Current checkpoint: v1 and the v2 negative pre-test result are committed at `503ee9a`.
V2.1 starts with a causal near-miss audit. Test images, labels, normalization values, and
metrics remain sealed until one newly frozen candidate passes every validation gate.

## 2. Problem

The v2 candidate with focus weight `0.03` came within the segmentation tolerance and
passed focus direction/MAE, but did not beat naive superposition on focus MAE. Candidates
with stronger focus did beat superposition but exceeded the segmentation drop. The
failure may arise from the six-mode basis, scalar-weight optimization, a soft-loss versus
hard-PQ mismatch, or a noisy small hard-validation screen.

V2.1 must distinguish those causes using training/validation evidence before changing
the model, then evaluate only targeted interventions without weakening any gate.

## 3. Proposed solution

1. Audit every existing v2 candidate by well, depth, density, hard metric component,
   focus error, soft objective, and local exact-gradient geometry.
2. Test a basis ladder: B6 control, B7 with Noll 11 primary spherical, then a compact
   fourth-order extension only if B7 evidence warrants it. Preserve the 2.5-radian RMS
   ball and exclude piston, tilt, and free defocus.
3. Implement an exact-gradient epsilon-constraint or augmented-Lagrangian continuation
   anchored at the matched segmentation-only pupil, with multiple deterministic starts.
4. Quantify InstanSeg raw-head component alignment with hard PQ and introduce only the
   smallest frozen v2.1 reweighting supported by validation evidence.
5. Promote candidates to a substantially larger well-grouped validation screen with
   paired well bootstrap intervals and deterministic site/crop selection.
6. Freeze one candidate, code commit, hashes, and protocol only if every gate passes;
   then run the locked test exactly once and never tune from it.
7. Produce photon, quantization, mismatch, robustness, decision-loop, and judge-facing
   evidence only after an eligible promotion.

## 4. Files to change

Existing v1 implementation files remain untouched wherever possible. V2 uses:

- `experiments/v1/`: immutable v1 freeze manifest and continuity note.
- `configs/v2/`: data, calibration, optics, objectives, designs, and pre-test freeze.
- `data/manifests/v2/`: BBBC006 provenance, checksums, well splits, and registrations.
- `src/tessscope/v2/`: BBBC006, calibration, autofocus, optics adapters, joint loss,
  optimization, and evaluation.
- `services/v2/`: SciPy autofocus plus any v2-specific optics/observer Tesseracts.
- `scripts/v2_*.py`: reproducible staged v2 commands.
- `tests/v2/`: unit, adjoint, integration, statistical, and smoke tests.
- `artifacts/runs/v2/`: generated calibration, derivative, optimization, and test evidence.
- `outputs/v2/`: final user-facing results, figures, demo material, and reproduction guide.
- `configs/v2_1/`, `artifacts/runs/v2_1/`, `outputs/v2_1/`, and `v2_1`-prefixed code:
  separately named continuation contract, audit, optimization, and evidence.
- Root brief, plan, decision log, README, and notices: current v2 status and navigation.

## 5. Step by step tasks

### V2.1 active work

- [x] V2.1-0A: preserve the verified v1/v2 state in local commit `503ee9a` after ignore,
  secret, size, Ruff, and 78-test checks.
- [x] V2.1-1A: created a machine-readable near-miss audit across all existing designs.
- [x] V2.1-1B: decomposed failure by well, depth, field density, PQ components, focus error,
  stage correction, and soft/hard correlation.
- [x] V2.1-1C: inspected coefficient paths, branch gradients, local exact-gradient sweeps,
  and classify the main blocker before selecting an intervention.
- [x] V2.1-2A: implemented and numerically validated B6/B7 basis conventions and RMS mapping.
- [x] V2.1-2B: passed support-energy, smoothness, quantization, and served derivative gates
  for every promoted basis.
- [x] V2.1-2C: extended B7 to compact B11 Noll modes 5–15, then passed basis,
  support-energy, quantization, and full served derivative gates under the same RMS ball.
- [x] V2.1-3A: implemented exact-gradient SLSQP epsilon-constraint continuation with
  shared exact-evaluation caching, deterministic starts, and explicit feasibility checks.
- [ ] V2.1-3B: measure runtime, screen cheaply, and compare multiple converged starts with
  weighted sums, PCGrad, naive superposition, and matched gradient-free search.
- [x] V2.1-4A: measured raw-head loss-component correlation with hard PQ/RQ/Dice/count.
- [x] V2.1-4B: retained the frozen differentiable loss because its validation correlation
  with hard PQ is already strong; no loss amendment or new ablation is warranted.
- [x] V2.1-5A: expanded B7 hard validation to the maximum 45/48 valid deterministic wells,
  recorded all three frozen-quality exclusions, and calculated paired well bootstraps.
- [ ] V2.1-5B: promote only a candidate that passes every unchanged gate without dependence
  on one field or start.
- [ ] V2.1-6A: if eligible, freeze and commit the v2.1 pre-test manifest before any test data.
- [ ] V2.1-6B: if eligible, run the locked test exactly once and preserve its result.
- [ ] V2.1-7A: produce robustness, figures, animation, reproduction, and submission evidence
  only after promotion.

### Frozen v2 history

- [x] V2-0A: inspect the repository and user work before editing.
- [x] V2-0B: hash 98 pre-v2 files into immutable `experiments/v1/freeze.json`.
- [x] V2-0C: checkpoint the approved v2 question, architecture, gates, and execution plan.
- [x] V2-1A: verify official BBBC006 metadata, licenses, z/wavelength/sampling semantics,
  and the exact download/extraction route.
- [x] V2-1B: implement bounded Hoechst-only acquisition with archive/file checksums.
- [x] V2-1C: implement well/site/plane parsing and freeze deterministic well splits before
  any test metric access.
- [x] V2-1D: register z13–z19 stacks to z16 and audit residual alignment on train/validation.
- [x] V2-1E: locate/decode automated reference masks and record their claim limitations.
- [x] V2-2A: define fixed spectral normalization and approximately 10×12 radial/angular
  Fourier features using training-only constants.
- [x] V2-2B: implement ridge fit/apply and signed-z/stage outputs.
- [x] V2-2C: derive and implement the analytic feature VJP and implicit ridge VJP without
  wrapping another autograd system.
- [x] V2-2D: pass small-array, random-direction, and served SciPy-Tesseract derivative tests.
- [x] V2-2E: validate clear and classical-astigmatic focus feasibility on train/validation.
- [x] V2-3A: estimate training-only OTF/transfer ratios and fit a bounded clear-microscope
  calibration with fixed aberration/background/noise parameters.
- [x] V2-3B: pass the predeclared held-out validation calibration gate before test access.
- [x] V2-3C: implement v2 phase basis/constraints, cubic and astigmatic baselines, true
  Poisson evaluation, and differentiable training noise approximation.
- [x] V2-3D: compose JAX optics, SciPy autofocus, and PyTorch InstanSeg branches into one
  objective and one pupil gradient.
- [x] V2-3E: pass every component derivative and the full served three-Tesseract gate.
- [x] V2-4A: benchmark M2 memory/runtime and freeze bounded optimization/evaluation budgets.
- [x] V2-4B: run smoke joint optimization and a small validation-only objective-weight
  Pareto sweep.
- [x] V2-4C: run matched clear, cubic, astigmatic, segmentation-only, focus-only,
  superposed, exact joint, broken-gradient, and derivative-free designs.
- [ ] V2-4D: multiple promoted joint starts completed; quantization/mismatch tests remain
  if the measured budget permits.
- [ ] V2-5A: blocked—no candidate passed all hard validation gates, so the frozen
  well-grouped test protocol was not promoted.
- [ ] V2-5B: blocked—the locked test was deliberately not accessed.
- [ ] V2-5C: blocked—the final test analysis and positive-or-negative locked claim require
  one eligible test run.
- [ ] V2-6A: generate the decision-loop panel, Pareto plot, gradient evidence, calibration
  residuals, pupil/PSFs, and optimization trace/animation.
- [ ] V2-6B: partially complete—checkpoint reproduction commands, limitations, and the
  hardware-ready claim boundary are recorded; judge-facing final evidence is blocked.
- [x] V2-6C: ran full lint/tests/evidence audit and preserved the pre-test result without
  publishing, committing, or submitting.

## 6. Acceptance criteria

- V1 code/config/evidence hashes remain recorded, and all v2 artifacts are separately
  named.
- BBBC006 planes/sites never cross well splits; no test labels, images, normalization,
  or calibration values influence tuning.
- Autofocus feature and implicit-ridge VJPs pass independent finite differences.
- Full served three-Tesseract derivative has relative error `< 1e-2` and cosine `> 0.99`.
- Clear digital-twin validation reaches the frozen calibration threshold, initially
  depth-curve correlation `≥ 0.90`, or v2 stops before learned-pupil claims.
- Nonzero-depth signed direction is `≥ 90%`; signed-z MAE is `≤ 2 µm`.
- Joint off-focus hard-instance PQ is at least clear `+0.01` with well-grouped confidence,
  and no more than `0.01` below segmentation-only.
- Joint Pareto-dominates naive superposition and matched derivative-free search.
- One predicted correction reduces residual defocus and improves hard PQ.
- Photon means are positive at both frozen levels for a positive robustness claim.
- All baselines share exposure, throughput, phase family/bounds, crop, normalization,
  data, observer, starts, and fair function/wall-time budgets.
- The complete smoke path runs locally without global installs or hidden credentials.

## 7. Testing plan

- Provenance, filename parsing, well grouping, checksum, and deterministic split tests.
- Registration identity/known-shift tests and real-stack residual diagnostics.
- FFT/log-power/bin normalization forward tests with fixed reference arrays.
- Hand-derived feature VJP and implicit ridge VJP central finite differences in float64.
- Served autofocus Tesseract forward/VJP parity and full three-boundary random directions.
- Phase RMS, coefficient bounds, PSF support, quantization, exposure, and noise tests.
- Joint-objective branch weighting and cotangent-sum tests.
- Focus direction/MAE, stage correction, hard instance metrics, DOF, and well-bootstrap tests.
- Tiny end-to-end smoke optimization before every larger run.
- Runtime/memory projection before locked evaluation.
- Final `uv run ruff check .`, `uv run pytest`, freeze verification, row-count audit, and
  visual inspection of every promoted figure.

## 8. Open questions

These are bounded training/validation decisions, not reasons to pause implementation:

- Exact official BBBC006 archive granularity and whether required w1 planes can be
  streamed individually or must be filtered from per-plane archives.
- The most stable registration/OTF-ratio mask and calibration parameter subset supported
  by training data without overfitting.
- Whether the operational focus Tesseract refits ridge weights on simulated support
  images during every pupil evaluation or freezes training-derived weights after each
  promoted design. Implement and validate the stronger implicit-refit path first.
- Whether the near-miss is dominated by basis expressivity, optimization geometry,
  soft-to-hard loss mismatch, or the small promoted hard screen.
- Whether B7 supplies sufficient spherical EDOF freedom or a compact B11 extension is
  justified by validation evidence.
- Which exact constrained method is most stable under measured M2 runtime: an
  epsilon-constraint continuation, augmented Lagrangian, or SLSQP with exact gradients.
- How much expanded hard validation is feasible while retaining a sealed validation-final
  subset against repeated tuning.
