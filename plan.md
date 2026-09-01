# TessScope v2 implementation plan

## 1. Goal

Build and evaluate a BBBC006-calibrated three-Tesseract optical co-design prototype in
which one bounded phase-only pupil jointly preserves nucleus segmentation and predicts
signed defocus/stage action from one photon-matched fluorescence exposure.

The full v2 handoff is approved. Work proceeds autonomously unless a consequential
scientific choice cannot be resolved on training/validation data, or a destructive,
legal, credential, billing, publication, conflicting-user-work, or external blocker
requires the user.

Current checkpoint: implementation reached hard train/validation evaluation, but no
candidate satisfied every frozen compromise gate. Test access is intentionally blocked;
see `outputs/v2/STATUS.md` and `configs/v2/pretest-block.json`.

## 2. Problem

Segmentation-only optics and signed autofocus are scientifically conflicting objectives.
TessScope must prove that a joint exact gradient across JAX/Chromatix, a transparent
NumPy/SciPy analytic adjoint, and PyTorch/InstanSeg finds a better optical compromise
than independent tuning, naive phase superposition, a broken-gradient ablation, or a
matched derivative-free search.

V2 must also strengthen v1 with real z-stack calibration, well-level statistics, a real
stage-correction decision loop, and positive or honestly negative photon evidence.

## 3. Proposed solution

1. Freeze and hash the complete v1 implementation/evidence without modifying its
   scientific result.
2. Stream the required BBBC006 Hoechst planes, build well-grouped deterministic splits,
   register stacks, load reference masks, and freeze provenance.
3. Fit a training-only low-dimensional clear-microscope calibration and gate it on
   held-out validation z-stacks.
4. Implement a NumPy/SciPy spectral autofocus Tesseract with an analytic FFT/feature VJP
   and implicit ridge-solve VJP, each independently checked by finite differences.
5. Add separately named v2 optics and joint-objective interfaces; combine focus and
   segmentation cotangents in one served gradient.
6. Pass component and full three-Tesseract derivative gates before optimization.
7. Measure runtime, run bounded train/validation-only objective-weight selection, and
   promote matched baselines/designs.
8. Freeze all v2 decisions and hashes, then run one paired, well-grouped locked test.
9. Produce decision-loop figures, Pareto evidence, mismatch/quantization results,
   reproducible commands, limitations, and later submission polish.

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
- Root brief, plan, decision log, README, and notices: current v2 status and navigation.

## 5. Step by step tasks

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

These are bounded train/validation decisions, not reasons to pause implementation:

- Exact official BBBC006 archive granularity and whether required w1 planes can be
  streamed individually or must be filtered from per-plane archives.
- The most stable registration/OTF-ratio mask and calibration parameter subset supported
  by training data without overfitting.
- Whether the operational focus Tesseract refits ridge weights on simulated support
  images during every pupil evaluation or freezes training-derived weights after each
  promoted design. Implement and validate the stronger implicit-refit path first.
- Validation-only joint objective weights, phase mode count, step budget, ridge lambda,
  and quantization/mismatch priority within the measured M2 budget.
