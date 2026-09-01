# TessScope v2.4 frozen-checkpoint audit plan

## 1. Goal

Audit every saved intermediate checkpoint from the frozen v2.3 closed-loop optimization
under the unchanged soft segmentation ceiling. Select validation checkpoints by a
pre-registered rule without retraining, hard-label cherry-picking, or locked-test access.

The full v2 handoff is approved. Work proceeds autonomously unless a consequential
scientific choice cannot be resolved on training/validation data, or a destructive,
legal, credential, billing, publication, conflicting-user-work, or external blocker
requires the user.

Current checkpoint: v1 through v2.3 are preserved, and v2.3 is frozen at `93da502` as a
correct negative endpoint-only experiment. V2.4 is explicitly approved. Before any
intermediate validation value is computed, v2.4 freezes the source hash, all 270 saved
checkpoints, the unchanged `1.098270310640335` ceiling, selection/diversity rules, hard
gates, and conditional v2.5 continuation. The BBBC006 locked test remains sealed.

## 2. Problem

V2.3 selected only the final step of each fixed 30-step Adam trajectory. All three starts
met the frozen first-frame segmentation ceiling, all nine endpoints improved residual
defocus, and all 270 intermediate parameter vectors were saved before intermediate
validation selection existed. An earlier checkpoint may therefore improve closed-loop
focus while still satisfying the unchanged first-frame constraint.

V2.4 must test that hypothesis over the frozen candidate pool without rerunning training
or using hard labels to select a step. If no saved checkpoint qualifies, the separately
authorized v2.5 experiment will use transparent exact-gradient constrained optimization.

### Historical v2.2 problem

V2.1's frozen `b7-projected-0.50-step-30` candidate gains `+0.03995` hard PQ over clear,
stays only `0.00642` below B7 segmentation-only, reaches `98.89%` direction and
`1.21073 µm` MAE, and improves hard PQ after one stage action. Its only failed v2.1 gate
is strict dominance over a single focus-heavy naive sum with much lower PQ. Two
non-dominating points do not define a fair multi-objective frontier.

V2.2 must determine whether the frozen joint candidate lies beyond a generously sampled,
physically matched family made only from the frozen B7 segmentation and focus pupils.
The joint pupil cannot be tuned during this comparison.

## 3. Proposed solution

1. Hash the frozen v2.3 matrix and create an integrity manifest containing exactly 9 runs
   × 30 steps, deterministic checkpoint identifiers, parameter hashes, and no new metrics.
2. Pre-register eligibility, three-objective nondominance, the existing tie-break, and a
   diversity rule allowing at most one selected checkpoint per v2.3 run.
3. Reconstruct all 270 frozen parameter vectors and evaluate the unchanged exact soft
   pipeline on the same 12 validation wells, caching exact parameter-hash duplicates.
4. Preserve every metric row and trajectory. Select at most three candidates only from
   soft-eligible nondominated checkpoints.
5. If any qualify, freeze selection before stopped-stage and expanded hard evaluation;
   apply all unchanged v2.3 hard gates before any derivative-free or locked-test work.
6. If none qualify, freeze v2.4 negative and continue automatically into v2.5 constrained
   closed-loop optimization without relaxing the first-frame ceiling.

### Historical v2.2 solution

1. Freeze the exact B7 joint candidate and pre-register the mixture equations, grids,
   projection, soft-screen rule, matched-frontier effects, uncertainty, and hypervolume
   reference before computing any v2.2 frontier result.
2. Generate convex interpolation, focus injection, nonnegative two-weight mixtures, and
   the original naive sum from the matched B7 separate pupils. Project only vectors that
   exceed the open 2.5-radian RMS ball; retain interior amplitude points.
3. Use the unchanged 12-well served soft endpoint only to remove clearly dominated
   mixtures, then evaluate every potentially relevant point on the frozen 45-well hard
   protocol with 27 hard-density wells and paired well bootstrap uncertainty.
4. Compare the frozen joint pupil with the piecewise envelope at matched segmentation
   and matched focus, including one predicted stage correction and application-facing
   usable-frame evidence.
5. If every unchanged gate plus the predeclared frontier gap passes, freeze and commit
   the candidate before exactly one locked test. If it fails, preserve v2.2 and continue
   as separately named v2.3 closed-loop differentiable microscope work.

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
- `configs/v2_2/`, `artifacts/runs/v2_2/`, `outputs/v2_2/`, and `v2_2`-prefixed code:
  frozen joint-candidate manifest, piecewise frontier contract, screening, hard metrics,
  uncertainty, and decision evidence.
- If required by a negative v2.2 decision, equivalent `v2_3` namespaces hold the
  separately approved closed-loop feedback experiment.
- `configs/v2_4/`, `artifacts/runs/v2_4/`, `outputs/v2_4/`, `src/tessscope/v2_4/`, and
  `v2_4`-prefixed scripts/tests hold the frozen-checkpoint audit without modifying v2.3.
- If v2.4 has no eligible checkpoint, equivalent `v2_5` namespaces hold the separately
  authorized constrained closed-loop optimization.
- Root brief, plan, decision log, README, and notices: current v2 status and navigation.

## 5. Step by step tasks

### V2.4 active frozen-checkpoint audit

- [x] V2.4-0A: froze the v2.3 matrix hash, exact 270-checkpoint pool, unchanged ceiling,
  eligibility, nondominance, diversity, hard gates, downstream order, and test policy.
- [x] V2.4-0B: added and passed an integrity test proving the pool contains exactly the
  expected frozen run/step identifiers and parameter hashes without validation metrics.
- [x] V2.4-1A: evaluated all 269 unique parameter hashes and preserved all 270 source
  rows on the same four exact soft validation batches. Endpoint reproduction error was
  below `8.5e-8`; no training, hard labels, or test data were accessed.
- [x] V2.4-1B: generated nine trajectory and two tradeoff figures. Ninety checkpoints
  were eligible and 57 were nondominated; the frozen one-per-run rule selected balanced
  step 14, action-heavy step 12, and first-heavy step 9 from segmentation-only starts.
- [ ] V2.4-2A: if selected, run matched stopped-stage comparisons and the unchanged
  expanded hard protocol; otherwise freeze a negative audit and activate v2.5.
- [ ] V2.4-2B: only after every earlier hard gate, run matched derivative-free control,
  freeze a verified local pre-test commit, and access the locked test exactly once.

### Conditional V2.5 constrained continuation

- [ ] V2.5-0A: activate only if v2.4 has no eligible checkpoint; pre-register a transparent
  exact-gradient constrained optimizer with validation checkpointing from the outset.
- [ ] V2.5-1A: verify objective and constraint gradients, run the bounded frozen-start
  matrix, and apply the same stopped-stage, hard, derivative-free, and locked-test order.

### V2.2 completed matched-frontier work

- [x] V2.2-0A: preserve v2.1 at local commit `4a88bdd` and keep every prior result
  immutable and separately named.
- [x] V2.2-0B: pre-register the frozen B7 joint candidate, source hashes, mixture family,
  physical projection, screening rule, matched effects, uncertainty, and test policy.
- [x] V2.2-1A: implemented deterministic mixture generation, deduplication, RMS projection,
  support/manufacturability checks, and unit tests.
- [x] V2.2-1B: ran the unchanged served soft endpoint for all 87 unique points and froze
  38 potentially relevant hard-frontier points without using hard labels.
- [x] V2.2-2A: evaluated all 38 frozen piecewise points on all 45 valid validation wells,
  including RQ/SQ/Dice/count, focus, photons, and one-step correction metrics.
- [x] V2.2-2B: calculated paired well uncertainty and the predeclared matched-segmentation,
  matched-focus, and normalized hypervolume comparisons.
- [x] V2.2-3A: applied every promotion gate. Piecewise-028 slightly dominates the frozen
  joint candidate, so v2.2 fails the matched-frontier gate without test access.
- [x] V2.2-3B: froze the negative v2.2 result and activated the approved v2.3
  feedback-loop experiment. No matched gradient-free or locked-test run was warranted.

### V2.3 completed closed-loop work

- [x] V2.3-0A: pre-registered the closed-loop objective, bounded stage action, first/final
  segmentation terms, transparent penalties, comparison baselines, and promotion gates.
- [x] V2.3-1A: implemented one differentiable loop using the same pupil for first and second
  exposures and propagate gradients through predicted stage action and residual depth.
- [x] V2.3-1B: added a forward-identical stopped-stage-gradient ablation and passed the full
  feedback-loop directional derivative gate.
- [x] V2.3-2A: completed all nine pre-registered exact profile/start runs (270 Adam steps).
  Every endpoint reduced residual MAE, but all nine exceeded the protected first-frame
  segmentation-loss ceiling; the nearest miss was `+0.002099`.
- [x] V2.3-2B: applied the frozen promotion order. With zero soft-eligible endpoints,
  stopped-stage optimization, expanded hard validation, matched derivative-free control,
  and locked-test access were correctly skipped. The negative result and evidence hashes
  are frozen in `configs/v2_3/pretest-block.json`.

### V2.1 completed negative pre-test checkpoint

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
- [x] V2.1-3B: measured runtime, screened multiple exact SLSQP starts, and compared
  weighted-sum history, segmentation-primary projected gradients, separate optima, and
  naive superposition. The matched B11 derivative-free run was conditionally skipped
  because no B11 candidate cleared the earlier segmentation and superposition gates.
- [x] V2.1-4A: measured raw-head loss-component correlation with hard PQ/RQ/Dice/count.
- [x] V2.1-4B: retained the frozen differentiable loss because its validation correlation
  with hard PQ is already strong; no loss amendment or new ablation is warranted.
- [x] V2.1-5A: expanded B7 and B11 hard validation to the maximum 45/48 valid deterministic
  wells, recorded all three frozen-quality exclusions, and calculated paired well
  bootstraps.
- [x] V2.1-5B: applied every unchanged promotion gate. No B7 or B11 candidate passed them
  all, so no pupil was promoted.
- [x] V2.1-6A: froze the negative pre-test block and evidence hashes; no candidate freeze
  was created because none was eligible.
- [x] V2.1-6B: preserved the locked test without access because the prerequisite promotion
  failed.
- [x] V2.1-7A: preserved the audit, status, and reproduction evidence. Promotion-only
  robustness, animation, and submission claims were intentionally not produced.

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
- For v2.2, replace the v2.1 single-superposition comparison only in the new contract:
  the frozen joint pupil must improve focus MAE by at least `0.10 µm` at equal-or-better
  hard PQ, or improve hard PQ by at least `0.005` at equal-or-better focus MAE, with a
  positive paired well-bootstrap lower bound for the matched gap.
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
- Mixture-grid completeness, exact naive-sum inclusion, interior-point retention, open-ball
  projection, coefficient deduplication, soft Pareto retention, and matched-frontier tests.
- Tiny end-to-end smoke optimization before every larger run.
- Runtime/memory projection before locked evaluation.
- Final `uv run ruff check .`, `uv run pytest`, freeze verification, row-count audit, and
  visual inspection of every promoted figure.

## 8. Open questions

There is no unresolved question inside the approved v2.2/v2.3 scope. Both experiments
are complete negative validation results, and the locked test remains sealed. Any v2.4
intervention would be a new scientific scope requiring a separate plan and approval; it
must not be inferred from the completed fallback authorization.
