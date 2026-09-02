# TessScope evidence-first visual and judge-demo plan

## 1. Goal

Turn the frozen v1–v2.6 evidence into the strongest scientifically honest visual story
and lightweight judge-facing replay experience. The protected claim is: “TessScope uses
exact gradients across JAX/Chromatix, NumPy/SciPy autofocus, and PyTorch/InstanSeg to
causally improve closed-loop microscope correction over a forward-identical
stopped-gradient system.” V2.4 is the primary positive validation evidence; its failure
to significantly beat piecewise-028, the negative v2.5/v2.6 follow-ups, the sealed
BBBC006 test, and the absence of physical microscope validation remain prominent.

The demo must run from a small cached validation evidence pack without downloading the
full dataset, loading locked-test data, starting differentiable services, or retraining.

Determine, without repeating v2.5's infeasible pupil-only search, whether the frozen
TessScope near miss can earn a statistically credible corrected-PQ advantage over the
strongest fair piecewise baseline. V2.6 first measures the controller, pupil/observer,
and exposure ceilings. It then activates only the intervention supported by the frozen
diagnostic. V1–v2.5 remain immutable and the locked BBBC006 test stays sealed until every
preregistered prerequisite passes.

The ordered primary targets are: corrected PQ at least matched baseline `+0.005` with a
positive grouped lower bound, at least 60% positive wells, and positive leave-one-well-out
means; then focus MAE at most `1.0 µm`; then at least 85% corrected hard frames improved.
The practical corrected-PQ target is `0.56`, with `0.60` a stretch target.

### Historical v2.5 goal

Test whether exact constrained optimization of the corrected frame in a manufacturable
B11 pupil can convert the frozen v2.4 hard near miss into a result that passes every
unchanged segmentation, autofocus, causal-gradient, piecewise, and stability gate.

The full v2 handoff is approved. Work proceeds autonomously unless a consequential
scientific choice cannot be resolved on training/validation data, or a destructive,
legal, credential, billing, publication, conflicting-user-work, or external blocker
requires the user.

Current checkpoint: v1 through v2.4 are preserved at local commit `b82000d`. V2.4 is a
complete negative expanded-hard result, but its best exact checkpoint missed the B7
segmentation tolerance by only `0.0002953` and beat its matched stopped-stage control by
`+0.006259` with a positive paired interval. The user explicitly approved a new v2.5
experiment on this hard near miss. V2.5 is now complete as a negative training/validation
experiment: zero B11 and zero B7 candidates passed the frozen constrained soft gates.
The locked BBBC006 test remains sealed.

## 2. Problem

The repository contains strong machine-readable evidence, but its existing top-level
figures emphasize the earlier v1 locked test and one qualitative test example. A judge
cannot yet see the v2.4 closed-loop causal result, inspect a traceable validation crop,
or distinguish cached replay from live computation in one concise experience. New
visuals must be generated from permitted validation data with fixed display rules and a
frozen representative-example selection—not chosen for presentation appeal.

The frozen v2.4 balanced checkpoint has a real causal corrected-PQ gain over its
forward-identical stopped-stage control, but it does not beat piecewise-028 by the frozen
effect, confidence, or stability requirements. V2.5 showed that more focus pressure and
B11 capacity can lower residual MAE while violating first-frame segmentation. The next
step must identify the actual ceiling before choosing controller calibration, exposure
allocation, more pupil capacity, or a physically defensible active-acquisition route.

### Historical v2.5 problem

Fixed weighted Adam improved residual focus but did not directly enforce the first-frame
and residual constraints. Early stopping came close on hard validation, so the next
scientific question is whether a transparent constrained method and four additional
manufacturable B11 modes can reach the small missing region without relaxing a gate.

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

1. Freeze an evidence claim matrix and immutable traceability manifest before selecting
   examples or rendering figures.
2. Select validation examples with a deterministic evidence-only rule, then build a
   compact cached sample pack containing the same crop, geometry, labels, predictions,
   depth, designs, display window, and source hashes used by every qualitative visual.
3. Generate high-resolution PNG plus SVG/PDF scientific figures for microscopy,
   pupil/PSF, depth curves, causal exact-versus-stopped evidence, and architecture.
4. Generate a broadly playable depth-sweep animation only if the permitted validation
   data provide all required frames under the same fixed visual transform.
5. Build a local static judge demo whose first view shows the causal closed-loop result,
   whose interactions replay cached evidence, and whose copy explicitly separates
   validated findings, failed comparisons, negative follow-ups, and unvalidated claims.
6. Verify hashes, no-test access, deterministic regeneration, accessibility, responsive
   layout, publication dimensions, one-command startup, lint, and the full test suite.

1. Freeze v2.4/v2.5 evidence hashes, a training-only 192/48/48 well-grouped
   optimization/development/confirmation partition, the oracle definitions, controller
   family, exposure grid, uncertainty, and the intervention decision rule.
2. Reconstruct the current balanced and piecewise field-level PQ-versus-depth curves.
   Measure exact focus/oracle ceilings, controller residual error by depth/density/well,
   and bounded gain/bias/monotone calibration headroom without test access.
3. Measure first/second exposure allocation on frozen training development/confirmation
   patches under a fixed 400-photon two-exposure budget and fixed noise seeds.
4. Activate joint B7 pupil/controller/exposure optimization only if the confirmation
   diagnostic reaches the minimum corrected-PQ target with first-frame preservation.
   Use feasibility restoration before corrected-task optimization and verify every new
   gradient independently.
5. If controller/exposure cannot reach the target, freeze that result and research the
   physical plausibility of rapid sequential SLM masks. Activate a separately
   preregistered two-mask design only if it is defensible and can be compared against a
   matched two-mask piecewise baseline through end-to-end gradients.
6. Freeze a terminal positive or negative v2.6 decision. Do not access the locked test
   without one frozen candidate and every applicable fair-baseline/causal gate.

### Historical v2.5 solution

1. Freeze source artifacts, anchors, exact B7→B11 lifts, optimizer budgets, epsilon ladder,
   alternate schedule, comparisons, derivative gates, and the single locked-test policy.
2. Add a differentiable B11 closed-loop optics service and validate the full exact
   derivative, including a nontrivial stage-path contribution.
3. Run the fixed-budget exact augmented-Lagrangian ladder for B7 continuity and B11
   primary starts. If a basis has no eligible endpoint, run the preregistered SLSQP
   alternate without changing constraints.
4. Evaluate every endpoint on the same 12 validation wells; freeze at most three feasible,
   residual-improving, nondominated B11 candidates with start/hash diversity.
5. Build and soft-screen the full B11 segmentation/focus piecewise family, then freeze its
   hard comparison set without using hard labels.
6. Run matched stopped-stage controls and the unchanged 45/27-well hard protocol. Run a
   matched derivative-free control only if every earlier hard gate passes.
7. Open the locked test exactly once only after all gates, hashes, checks, and a local
   pretest commit pass. Otherwise freeze a complete negative v2.5 result.

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
- `configs/v2_5/`, `artifacts/runs/v2_5/`, `outputs/v2_5/`, `src/tessscope/v2_5/`,
  `services/v2_5/`, and `v2_5`-prefixed scripts/tests hold the separately authorized
  hard-near-miss constrained B11 experiment without modifying v1–v2.4.
- `configs/v2_6/`, `data/manifests/v2_6/`, `artifacts/runs/v2_6/`, `outputs/v2_6/`,
  `src/tessscope/v2_6/`, `services/v2_6/`, and `v2_6`-prefixed scripts/tests hold the
  diagnosis-first controller/exposure or conditional active-acquisition experiment.
- Root brief, plan, decision log, README, and notices: current v2 status and navigation.
- `configs/demo/`: frozen claim matrix, example-selection rule, visual contract, and
  evidence manifest.
- `artifacts/runs/demo/`: deterministic traceability, selection, cached validation sample,
  and figure-generation audit artifacts.
- `outputs/demo/`: publication figures, captions/alt text, animation, and the local static
  judge experience.
- `scripts/build_demo_evidence.py`, `scripts/generate_demo_figures.py`, and
  `scripts/serve_demo.py`: one-way evidence preparation, deterministic rendering, and
  one-command local replay.
- `tests/demo/`: claim, traceability, no-test-access, selection, output, and demo checks.

## 5. Step by step tasks

### Evidence-first visual and judge-demo phase

- [x] DEMO-0A: received explicit authorization, confirmed a clean repository at
  `ccec9b2`, and preserved v1–v2.6 without push, publication, deployment, deletion, or
  locked-test access.
- [x] DEMO-0B: freeze the claim matrix, source hashes, allowed-data boundary, global
  display transform, representative-example selection rule, output formats, and tests.
- [x] DEMO-1A: build and verify the immutable traceability manifest and select the
  representative validation example without presentation-quality cherry-picking.
- [x] DEMO-1B: materialize the minimal cached validation sample pack with real sensor
  frames, InstanSeg instances, reference labels, stage actions, and per-frame metrics.
- [x] DEMO-2A: generate the matched microscopy, pupil/PSF, depth-curve, causal-gradient,
  and architecture figures in publication and presentation formats.
- [x] DEMO-2B: generate a deterministic validation depth-sweep animation if complete
  matched frames are available; otherwise freeze a documented skip.
- [x] DEMO-3A: build the one-command local judge replay with explicit cached/live status,
  concise claims, limitations, captions, alt text, and accessible interactions.
- [x] DEMO-3B: verify publication sizes, deterministic hashes, local startup, responsive
  layout, no-test access, Ruff, and the full suite; freeze milestone commits.

### V2.6 diagnosis-first improvement

- [x] V2.6-0A: received explicit authorization, preserved v1–v2.5 at local commit
  `09b9eb2`, and defined a new v2.6 namespace with no push/deploy/submission authority.
- [x] V2.6-0B: froze source hashes, training-only well partitions, oracle definitions,
  controller/exposure grids, decision rules, targets, matched baselines, and test seal.
- [x] V2.6-1A: implemented pure oracle/PQ-curve/controller decomposition utilities and
  tests, including grouped uncertainty and leave-one-well-out stability.
- [x] V2.6-1B: ran the frozen balanced-versus-piecewise diagnostic and quantified controller
  error, optical/observer ceiling, depth/density/well structure, saturation, bias, gain,
  and nonlinearity.
- [x] V2.6-1C: ran the fixed-total-photon exposure audit on frozen training-only
  development and confirmation patches with no validation/test tuning.
- [x] V2.6-2A: applied the frozen route decision, rejected bounded B7
  pupil/controller/exposure co-design, and froze the route as a negative result.
- [x] V2.6-2B: not activated—the oracle and exposure prerequisites failed before new
  controller/pupil generation; no derivative or expensive matrix was warranted.
- [x] V2.6-3A: not activated—the diagnosis rejected controller/exposure co-design before
  a new exact matrix or controller was generated.
- [x] V2.6-3B: not activated—no candidate cleared the diagnostic prerequisites for new
  hard validation or matched derivative-free comparison.
- [x] V2.6-4A: after rejecting the controller/exposure route, researched primary/official
  evidence for rapid sequential SLM masks and preregister a two-mask route only if it is
  physically and scientifically defensible.
- [x] V2.6-4B: froze the two-mask sources, B7 sensing/capture starts, training-only
  batches, matched 76-pair piecewise family, feasibility restoration, optimizer budget,
  selection, hard gates, and switching/photon limitations before optimization.
- [x] V2.6-4C: implemented the two-mask served graph and passed independent sensing, capture,
  residual-depth, exact/stopped parity, and full-loop derivative gates.
- [x] V2.6-4D: soft-screened all 76 matched two-mask piecewise pairs, ran four feasibility
  restorations and the bounded 72-step exact two-start matrix, and froze zero promotions.
- [x] V2.6-4E: not activated—zero baseline pair and zero exact endpoint passed the frozen
  residual/first-frame soft gates, so stopped-stage, hard, stability, and derivative-free
  work was not authorized.
- [x] V2.6-5A: froze the terminal v2.6 result, evidence hashes, limitations, and locked
  test decision; repository-wide Ruff and all 177 tests pass before the final milestone.

### V2.4 completed frozen-checkpoint audit

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
- [x] V2.4-2A: ran three matched stopped-stage controls with exact forward parity and
  evaluated all six frozen pupils on 45 wells/27 hard-density wells. No exact checkpoint
  passed the required corrected gain, paired confidence, and stability over piecewise-028.
- [x] V2.4-2B: applied the frozen ordering. The matched derivative-free control and
  locked test were correctly skipped because no checkpoint cleared the earlier hard gate;
  the negative pre-test evidence and hashes are frozen.

### V2.5 hard-near-miss constrained continuation

- [x] V2.5-0A: received explicit new authorization after the frozen v2.4 hard near miss;
  this does not rewrite the earlier, correctly inactive zero-soft-eligible trigger.
- [x] V2.5-0B: froze v2.4/B11 source hashes, B7 continuity and B11 start vectors, exact
  zero-padding, anchors, budgets, epsilon ladder, alternate method, comparisons, gates,
  and the locked-test seal before generating a new pupil.
- [x] V2.5-1A: implement and test exact augmented-Lagrangian/SLSQP constrained utilities,
  selection rules, B11 physical diagnostics, and the differentiable-depth B11 service.
- [x] V2.5-1B: passed the B11 full-loop derivative gate at `0.009022` relative error,
  `0.999887` cosine, `43.1%` stage-path gradient fraction, and exact forward parity.
- [x] V2.5-2A: completed the resumable B7 continuity and B11 primary constrained ladders:
  378 exact aggregate steps across 21 frozen endpoints.
- [x] V2.5-2B: activated the alternate schedule for both bases, completed six capped
  SLSQP runs, evaluated the 12-well soft set, and froze zero B11 candidates.
- [x] V2.5-3A: not activated—zero B11 candidate passed the prerequisite constrained soft
  screen, so the conditional matched piecewise family was correctly skipped.
- [x] V2.5-3B: not activated—zero B11 candidate was frozen for a stopped-stage control.
- [x] V2.5-4A: not activated—zero candidate passed the prerequisite training and soft
  gates, so no new hard labels were accessed.
- [x] V2.5-4B: not activated—the registered derivative-free and optional gain work was
  conditional on passing every earlier gate.
- [x] V2.5-5A: froze a complete negative pre-test result with zero promotions and kept the
  locked test sealed.
- [x] V2.5-5B: produced final integrity, audit, status, and reproduction evidence without
  push, deployment, publication, or submission.

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

### Evidence-first visual/demo criteria

- Every displayed number, crop, depth, design, prediction, pupil, PSF, and output maps to
  a source artifact/data record plus SHA-256 hash in one machine-readable manifest.
- Representative validation examples are chosen by a frozen deterministic rule using
  quantitative evidence fields only, before rendering or subjective inspection.
- Qualitative comparisons use identical crop geometry and a single documented global
  intensity window; no per-image normalization, sharpening, denoising, retouching, or
  invented cells are allowed.
- The core story displays v2.4 first PQ `0.492192`, corrected PQ `0.552295`, focus MAE
  `1.267923 µm`, signed direction `98.89%`, exact-minus-stopped `+0.00625884` with 95% CI
  `[0.0006703, 0.0115475]`, and the non-significant piecewise gap `+0.001285` with CI
  `[-0.001719, 0.004800]`. PQ is never presented as ordinary percent accuracy.
- The locked test remains sealed, no physical microscope validation is implied, and
  v2.5/v2.6 negative follow-ups remain visible without dominating the causal result.
- The judge experience starts with one command, uses only cached permitted evidence,
  works without the large dataset or retraining, and labels replay versus computation.
- Static outputs include high-resolution PNG and vector SVG/PDF where appropriate, with
  concise captions and alt text; animation is broadly playable when produced.

### V2.6 criteria

- Oracle and interpolated controller audits are paired at field/depth level and report
  grouped 95% well-bootstrap intervals, positive-well fraction, and leave-one-out means.
- Minimum system promotion is corrected PQ `>= matched baseline + 0.005`, positive paired
  lower bound, at least 60% positive wells, and every leave-one-well-out mean positive.
- First-frame hard PQ remains within `0.01` of matched segmentation-only and improves
  over clear by at least `0.01` with a positive paired lower bound.
- Focus MAE is at most `1.0 µm` for the primary target; `0.75 µm` is stretch only. Signed
  direction never falls below `90%`, with `99%` preferred.
- At least 85% of corrected hard frames improve for a positive v2.6 claim.
- Any exposure policy preserves a fixed total expected-photon budget and is given to the
  matched piecewise baseline with the same selection and evaluation budget.
- Any controller is bounded and interpretable: gain, bias, and at most one monotone cubic
  term, with action clipping preserved.
- New controller/pupil/exposure gradients pass central differences and full-loop relative
  error `<0.01`, cosine `>0.99`, and forward parity for stopped-stage controls.
- The locked test is accessed at most once only after a frozen candidate, controller,
  exposure policy, hashes, full checks, and a local pretest commit.

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

- Claim tests compare every headline number with its frozen v2.4 source value and reject
  percent-accuracy wording for PQ.
- Traceability tests verify every source/output hash, allowed split, field/crop/depth,
  design identity, display transform, and explicit `test_accessed: false` boundary.
- Selection tests rebuild the representative-example choice from evidence-only inputs and
  prove the selected well belongs to validation, never the locked test.
- Figure tests regenerate into a temporary directory, check deterministic hashes where
  supported, inspect PNG dimensions/modes, parse SVG, open PDF, and decode animation.
- Demo tests start the local server, request the page/assets, verify cached-replay labels,
  accessible image descriptions, core claims/limitations, and absence of remote data.

- V2.6 pure tests cover oracle focus mapping, interpolation bounds, controller monotonicity
  and clipping, grouped splits, uncertainty, decision ordering, and exposure conservation.
- Diagnostic integration tests assert exact source hashes, row pairing, hard-density
  counts, no test wells, frozen candidate identity, and deterministic output.
- Any activated served route receives controller/exposure component finite differences,
  full-loop directional derivatives, exact/stopped forward parity, and a small smoke run
  before an expensive matrix.

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

The visual/demo implementation question is resolved. Frozen metric rows were sufficient
for representative selection; the selected validation field was then recomputed once
with the frozen models/designs and cached as a deterministic 12.4 MiB replay. No locked
test record was read.

The v2.6 empirical question is resolved negatively. Neither bounded controller/exposure
calibration nor the separately preregistered sequential two-mask route reached the frozen
training-only feasibility region. No arbitrary fallback project is authorized, and the
locked test remains sealed.
