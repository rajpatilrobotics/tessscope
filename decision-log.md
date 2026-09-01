# TessScope decision log

## 2026-09-01 — V2.1 validation-only continuation authorized

- The verified v1/v2 state was preserved in local root commit `503ee9a`; downloaded
  datasets, third-party model weights, environments, caches, runtime output, and binary
  feature caches remain ignored. Compact JSON/CSV/PNG evidence is tracked.
- V2.1 is separately named and motivated by the frozen v2 near-miss. It may use existing
  training/validation artifacts but may not overwrite or relabel v2 evidence.
- The intervention order is causal audit, B7 primary-spherical basis extension, exact
  constrained continuation, evidence-supported soft-loss alignment, and expanded
  well-grouped validation. B11 is conditional on B7 evidence.
- The scientific gates are unchanged. Test images, labels, normalization values, and
  metrics remain sealed until one candidate passes every gate and a pre-test commit and
  hash manifest exist.
- Verified local milestone commits are authorized. Push, publication, deployment,
  destructive Git operations, test leakage, and post-hoc threshold changes remain
  prohibited.

## 2026-09-01 — V2.1 near-miss audit selects constrained B7 intervention

- Recomputed seed, instance-membership, and total differentiable InstanSeg losses for
  all 18 promoted v2 designs on validation-only fields. Total task loss strongly tracks
  hard dense off-focus PQ (Pearson `-0.9681`, Spearman `-0.9705`); instance loss alone
  reaches `-0.9592`/`-0.9581`. Loss reweighting is therefore not the first intervention.
- The exact branch-gradient audit found conflicts at 9/11 sampled points along the B6
  design path. Around the `0.03` near-miss, branch-gradient cosine was about `-0.62`;
  the observed range was `-0.7884` to `+0.7514`. Scalar weights are poorly matched to
  this local non-convex geometry.
- The `0.03` candidate beats naive superposition on hard dense PQ by `+0.02724` but has
  `0.17571 µm` higher focus MAE. After one correction it reaches hard PQ `0.41667`
  versus superposition `0.41407`; both correction loops improve PQ.
- Leave-one-field-out hard PQ for the near-miss spans `0.32855` to `0.41485`, confirming
  that the six-dense-field screen is too small for promotion even though focus metrics
  are stable. Expanded well-grouped validation remains mandatory.
- Decision: retain the well-aligned frozen loss, add B7 primary spherical under the same
  RMS ball, and use exact-gradient constrained continuation. This follows
  [Noll's unit-disk orthogonal convention](https://opg.optica.org/josa/abstract.cfm?uri=josa-66-3-207)
  and [SciPy's documented SLSQP interface](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
  for callable objective and constraint Jacobians. B11 remains conditional.

## 2026-09-01 — B7 screen promotes one start to expanded training

- A seven-point primary-spherical sweep across five B6 seeds found 11 nonzero B7 probes
  that strictly improved both validation task loss and focus MSE relative to their
  zero-spherical controls. B7 is retained; B11 is not justified before B7 is exhausted.
- Matched 90-step exact-gradient B7 baselines used 36 training wells and 12 validation
  wells per design. The B7 segmentation-only validation loss/focus MSE is
  `1.09027/4.59833`; focus-only is `1.14826/2.24958`; constrained naive superposition is
  `1.12248/1.45912`.
- Four deterministic starts and two epsilon margins were screened with exact served
  SLSQP Jacobians. Each stage was bounded at six iterations and retains solver status,
  all endpoints, traces, exact-evaluation counts, and wall time; no non-converged result
  is represented as converged.
- Only `v2_exact_plus_spherical_0.25-margin-0.004` satisfies both the training constraint
  (slack `+0.0000822`) and the validation soft-segmentation allowance. Its validation
  focus MSE is `5.21696`, so it is only eligible for expanded training, not for a positive
  claim or test access.
- The promotion filter requires training constraint slack of at least `-1e-4` and
  validation segmentation loss no greater than the B7 segmentation anchor plus `0.008`.
  The next step uses the full 36-well training budget, followed by one expanded 48-well
  hard-validation run only if feasibility is retained. Test data remains sealed.

## 2026-09-01 — Exact B7 full-training continuation does not promote

- The sole screened start was continued on all 36 fixed training wells under the
  predeclared `0.004` and `0.008` epsilon margins. Each run used 12 noisy training
  batches, 10 SLSQP iterations, exact served branch Jacobians, and preserved every
  callback and solver status.
- Both endpoints are training-feasible within the frozen `1e-4` numerical tolerance.
  The strict endpoint has task loss/focus MSE `1.21486/7.07375`; the loose endpoint has
  `1.21884/5.60451`.
- Neither generalizes through the held-out soft-segmentation allowance. Strict validation
  task loss is `1.10271` and loose is `1.11169`, versus the unchanged maximum `1.09827`.
  Their validation focus MSE values are `2.24339` and `2.34570`, respectively.
- Validation-based early stopping checked all nine training-feasible SLSQP callbacks.
  None passed the same held-out allowance; the best validation task loss was `1.10242`.
- Decision: do not promote an SLSQP endpoint or checkpoint to the 48-well hard screen.
  Continue with the already planned segmentation-primary projected exact-gradient
  comparison, which targets the measured branch conflict without changing the loss or
  success gates. Test data remains sealed.

## 2026-09-01 — Projected B7 gradients produce two hard-screen candidates

- Segmentation-primary projection removed only the normalized-focus gradient component
  opposing the exact segmentation gradient, then norm-balanced the remaining focus
  direction. Three balances used matched 30-step Adam schedules from the B7
  segmentation-only pupil on 36 training wells.
- All three balance-family winners pass the full-training and 12-well soft-validation
  segmentation constraints. Balance `1.0` step 30 reaches validation task loss/focus
  MSE `1.09695/2.19079`; balance `0.5` reaches `1.09611/2.52832`; balance `0.25`
  reaches `1.09327/3.03089`. The unchanged validation task-loss maximum is `1.09827`.
- The two lowest-focus eligible family winners, balances `1.0` and `0.5`, are frozen for
  expanded 48-well hard validation. Selection occurred before that hard run; no hard
  labels or metrics from the expanded fields influenced their coefficients.
- This promotion is validation eligibility only. Direction, MAE, hard dense PQ,
  stage-correction, naive-superposition, derivative-free, and uncertainty gates remain
  unproven. Test data remains sealed.

## 2026-09-01 — Expanded B7 hard validation isolates one remaining gate

- The frozen patch collector can supply one registered, supervised patch from 45 of the
  48 validation wells. Wells `l09` and `o20` have no field passing the frozen
  registration gate; `o13` has one registered field but no valid supervised patch
  instance. All exclusions are explicit; no test or substitute wells were used.
- The expanded matrix evaluates 45 distinct wells, seven depths, six frozen designs,
  official hard InstanSeg labels, candidate correction frames, and 2,000-replicate paired
  bootstraps grouped by well. The hard-density subset contains 27 wells.
- Projected balance `0.5` reaches hard dense off-focus PQ `0.49606`, a `+0.03995` gain
  over clear with 95% interval `[+0.01604, +0.06310]`, and only `0.00642` below matched
  B7 segmentation-only. Direction is `98.89%`, MAE is `1.21073 µm`, and correction
  improves hard PQ from `0.49606` to `0.55206`.
- Projected balance `1.0` reaches hard PQ `0.49440`, clear gain `+0.03829` with interval
  `[+0.01492, +0.06117]`, segmentation drop `0.00809`, direction `98.89%`, MAE
  `1.16210 µm`, and corrected hard PQ `0.55059`.
- Both candidates pass every preliminary unchanged gate except Pareto dominance over
  matched B7 naive superposition, whose hard PQ/MAE is `0.45401/0.90485 µm`. They have
  much higher PQ but do not beat its focus MAE. No candidate is promoted and test access
  remains blocked.
- Decision: the near-miss is now localized to basis expressivity rather than loss
  alignment, hard-PQ preservation, focus sign, stage action, photon positivity, or
  small-sample instability. Activate the plan's compact B11 fourth-order extension
  (Noll 5–15) under the same 2.5-radian RMS ball. A matched B11 derivative-free baseline
  is required only if a B11 candidate clears the naive-superposition gate.

## 2026-09-01 — Compact B11 basis and full derivative gates pass

- B11 retains Noll 5–11 and adds the four remaining radial-order-four modes, Noll 12–15,
  under the unchanged open 2.5-radian RMS coefficient ball. Piston, tilt, and free
  defocus remain excluded.
- On a 401×401 sampled unit disk, maximum basis diagonal error is `0.002893` and maximum
  off-diagonal magnitude is `0.001764`, both below `0.012`. Mixed and pure fourth-order
  probes retain at least `0.997224` PSF energy, above the unchanged `0.995` gate.
- Separate 11-parameter optics and observer services preserve the three-framework chain.
  The full JAX/Chromatix → NumPy/SciPy → PyTorch/InstanSeg derivative passes with median
  relative error `0.002532` and cosine `0.999446`.
- Decision: B11 is eligible for validation-only optimization. These numerical gates do
  not promote a pupil or unlock test data.

## 2026-09-01 — V2 stopped before test after a negative hard-validation gate

- The v2 implementation reached the complete training/validation checkpoint. No
  BBBC006 test-well image or label was loaded, and no locked test metric was run. A unit
  test exercises the manifest guard with `allow_test=True, require_files=False`; it
  reads frozen membership metadata only, not scientific test data.
- The first selected exact-joint pupil passed the clear gain, focus, stage-correction,
  superposition, and derivative-free comparisons. Its hard dense off-focus PQ was
  `0.35521`, versus `0.33660` for clear and `0.37894` for segmentation-only. The
  resulting `0.02373` drop from segmentation-only exceeded the frozen `0.01` limit.
- Scalar-weight refinements exposed the boundary rather than removing it. The
  joint-initialized `0.03` candidate met the segmentation-drop limit (`0.00979`),
  direction (`91.67%`), and MAE (`1.4399 µm`) gates, but did not Pareto-dominate naive
  superposition's `1.2642 µm` MAE. The `0.05` and `0.07` candidates beat superposition
  on focus but missed the segmentation-drop limit.
- Segmentation-primary projected exact branch gradients confirmed frequent negative
  focus/segmentation gradient dot products and preserved hard PQ, but none reached the
  signed-focus gate. This is retained as mechanistic evidence of the task conflict.
- Because no single candidate satisfied every frozen hard-validation condition, v2 test
  freeze/evaluation, photon claims, quantization/mismatch polish, and learned-pupil
  headline claims remain blocked. The correct current result is a negative pre-test
  finding, not a completed positive hackathon claim.

## 2026-09-01 — V2 calibration, autofocus, and three-Tesseract derivative gates passed

- All 5,376 requested BBBC006 Hoechst TIFFs and 768 automated z16 reference masks were
  range-extracted and checksum-validated. Deterministic well splits remain 288/48/48.
- Translation registration accepted 635/672 training/validation fields. Rejections were
  concentrated in low-content fields; accepted residual shift p95 was `0.2 px`.
- Training-only global intensity normalization is offset `117`, scale `2037`, with one
  global exposure gain `11.5820166`. Per-image normalization remains forbidden.
- Clear real-stack autofocus achieved `92.88%` direction accuracy and `1.6227 µm` MAE.
  The selected classical astigmatic design achieved `100%` and `0.3236 µm` on its
  bounded validation feasibility set.
- The zero-offset clear digital twin failed (`0.4121` validation depth-curve
  correlation) and was preserved. Adding one training-fitted axial offset (`-2.5 µm`)
  and depth scale (`0.95`) produced `0.95936` held-out validation correlation.
- The served NumPy/SciPy autofocus VJP passed with relative error `0.001066` and cosine
  `0.999610`. The unchanged full JAX optics → SciPy autofocus → PyTorch InstanSeg chain
  passed the v2 contract with relative error `0.007269` and cosine `0.999970` over two
  stable small-step epsilons.

## 2026-09-01 — TessScope v2 handoff approved and execution started

- The user approved the authoritative v2 handoff as the major implementation plan; no
  routine second approval is required after checkpointing the plan.
- TessScope remains the sole project. V2 asks whether one BBBC006-calibrated bounded
  phase pupil can preserve nucleus segmentation and predict signed stage action from
  one photon-matched fluorescence frame.
- V2 uses three scientifically load-bearing differentiation regimes: native JAX reverse
  mode in Chromatix optics, a transparent NumPy/SciPy analytic and implicit autofocus
  adjoint, and the frozen PyTorch/InstanSeg exact input VJP.
- The non-piecewise conflict is frozen: segmentation invariance destroys signed focus
  information, while strong focus coding can damage morphology. Independent pupils and
  their constrained naive superposition are required baselines.
- BBBC006 z13–z19 Hoechst stacks are the primary calibration/evaluation data. Splits and
  uncertainty are grouped by well. BBBC039 remains v1 continuity evidence only.
- The learned-mask claim is limited to BBBC006-calibrated in-silico/hardware-ready design;
  no fabrication or physical microscope validation is implied.
- M2 is the default platform. GPU use requires a measured blocker. Final eligibility,
  license presentation, Track 5 wording, and submission polish wait until scientific
  evidence is stable.

## 2026-09-01 — V1 frozen before v2 edits

- `scripts/v2_freeze_v1.py` recorded SHA256 values for 98 pre-v2 implementation,
  configuration, manifest, result, and evidence files in
  `experiments/v1/freeze.json`.
- Existing v1 modules/configs remain the original experiment. V2 additions use
  `src/tessscope/v2`, `services/v2`, `configs/v2`, `data/manifests/v2`,
  `artifacts/runs/v2`, `outputs/v2`, and `v2_`-prefixed scripts.
- The frozen v1 conclusion is unchanged: deterministic PQ improved, exact VJP beat the
  forward-identical surrogate, while count and photon headline gates failed.

## 2026-09-01 — Implementation authorized

- TessScope is the sole project.
- Contract v2 and the photon/headline/derivative amendments are frozen.
- Gate 0 observer access and parity precede data plumbing and optics implementation.
- Work proceeds autonomously after the user's “start implementation” approval, with a pause only for consequential scope/architecture changes, destructive actions, credentials/billing/publication, legal/licensing concerns, or conflicts with user work.
- The workspace was initially empty and was not a Git repository.

## Pending evidence-based decisions

- None before the one-start matched design runs.

## 2026-09-01 — Local runtime baseline

- Python 3.12 was selected because current Chromatix requires Python 3.12 or newer, while Python 3.12 is a conservative common target for PyTorch/InstanSeg and Tesseract-JAX on Apple Silicon.
- The official Chromatix install source is Git rather than a published `0.6.0` PyPI release, so it is pinned to commit `ce1482906fc663298613bb252bbf425e3be59839` instead of an unbounded `main` branch.
- The official InstanSeg model tag resolves to commit `9dcbf65bcc725d4e3121081f66c52d0a01e12579`; the matching package release tag `v0.1.1` resolves to `8f5f373bf72e7ae3a5037e9f19f2dd7d0ac3bc23`.

## 2026-09-01 — Gate 0 passed

- Official archive SHA256: `6bc2e4cd8acd9bea64a04e926a74a53f922d76fd6c1dc633ca0bca335df4d4d7`.
- The TorchScript exposes a frozen `fcn` and `pixel_classifier` directly. Its raw head has five ordered channels: two coordinate logits, two sigma channels, and one seed-distance logit.
- Release preprocessing is 0.1/99.9 percentile scaling without clipping. Clipping to `[0, 1]` is not equivalent and was rejected.
- The official test output is pixel-identical. Raw-head plus official source postprocessing yields the same 25 instance masks with zero differing pixels after deterministic first-pixel relabeling.
- Median 256×256 timings on this M2 host: CPU raw/VJP/official-hard `193.4/347.8/241.0 ms`; MPS `24.2/51.6/44.6 ms`.
- The eager source postprocessor has an upstream CPU/MPS tensor mismatch in `torch.isin`; parity therefore runs on CPU, while the bundled official hard endpoint and differentiable raw/VJP paths both run successfully on MPS.

## 2026-09-01 — Dataset decontamination frozen

- Official archive hashes, URLs, sizes, licenses, citations, and split counts are recorded in `data/manifests/bbbc-assets.json`.
- BBBC039 official split counts are 100 training, 50 validation, and 50 test; BBBC038 stage-1 contains 670 observer-training images.
- All 200×670 identity pairs were checked using file hashes, decoded grayscale hashes, normalized 64×64 D4 perceptual hashes, and maximum D4 normalized cross-correlation.
- Forty-three pairs exceeded the frozen NCC threshold of `0.995`; every pair also had pHash Hamming distance 0 and NCC between `0.9999222` and `1.0000006`.
- Identity-only side-by-side review confirmed all 43 as the same fields without showing masks, predictions, model scores, or image-quality judgments.
- Frozen exclusions: 25 training, 9 validation, and 9 test. The remaining official-split counts are 75/41/41; images are excluded in place and never reassigned between splits.

## 2026-09-01 — Optics support accepted

- The frozen 96-pixel sensor support at 4× oversampling retains between `0.9972982` and `0.9973172` of the energy in a same-sampling 2×-larger Chromatix reference across all seven clear-pupil depths.
- The minimum retained fraction exceeds the frozen `0.995` criterion, so the 128-pixel fallback is not activated.
- The implementation uses the pinned Chromatix commit for objective-point-source and Fourier-lens propagation, with JAX-native phase, integration, convolution, and VJP operations.

## 2026-09-01 — Gate 5 contract conflict (decision required)

- Both local HTTP Tesseracts serve successfully. A full JAX `value_and_grad` call crosses Chromatix/JAX → HTTP → InstanSeg/PyTorch → HTTP and returns a finite six-parameter gradient.
- The frozen observer says to sample the predicted embedding at each fixed GT center and detach that sampled value. This is a deliberate stop-gradient: the forward value still changes when that center embedding changes, while the VJP is explicitly defined to omit that path.
- Therefore an ordinary central finite difference and the frozen VJP are derivatives of different semantics. The required five-direction served gate fails at every epsilon; median relative errors are `0.4647`, `0.1384`, `0.1402`, and `0.2555`, with cosine agreements `0.6540`, `0.9163`, `0.9104`, and `0.8885` for epsilons `1e-1` through `1e-4`.
- Removing the detach only as a diagnostic moves the directional cosine above `0.99` at the three smaller epsilons, confirming the detached path is the dominant discrepancy, but it still does not justify changing the frozen contract without approval.
- Reproducible evidence is written to the ignored local artifact `artifacts/runs/gate5/served-directional-derivative.json`.

Resolution options, in recommended order:

1. Amend the design loss to use the connected predicted center embedding. This makes the VJP the derivative of the reported scalar objective and preserves the strongest scientific interpretation of the finite-difference gate, but changes the frozen detach rule.
2. Keep detach and redefine Gate 5 as a conditional finite difference with the center embeddings held fixed at the base point. This validates the intended surrogate VJP but is not the ordinary derivative of the complete reported forward objective.
3. Keep both clauses unchanged and accept Gate 5 as a documented negative result, which blocks optimization and headline experiments under the current acceptance criteria.

## 2026-09-01 — Connected-center amendment approved

- The user explicitly approved resolution option 1.
- The design loss now keeps the predicted embedding sampled at each fixed GT center connected to autograd. GT locations and crop indices remain fixed, while gradients flow through both the crop embeddings and their center references.
- This narrowly replaces the frozen detached-center clause so the served VJP can be the derivative of the scalar objective reported by the forward endpoint. All other loss definitions, thresholds, data rules, and budgets remain unchanged.

## 2026-09-01 — Gate 5 passed after the approved amendment

- Five deterministic Rademacher directions were evaluated at all frozen epsilons. The stable float32 window is `1e-2`, `1e-3`, and `1e-4`; the largest `1e-1` step is retained in the evidence as the expected nonlinear truncation regime.
- Full served JAX/Chromatix → PyTorch/InstanSeg result: overall median relative error `0.0052008`, cosine agreement `0.9996118`.
- Served optics component: median relative error `0.0027751`, cosine agreement `0.9997217` over the stable window.
- Served observer component on a linearized local-optics tangent basis: median relative error `0.0067493`, cosine agreement `0.9995660`.
- All three checks pass the required `<0.01` median relative error and `>0.99` cosine thresholds. None reaches the preferred `<0.001` relative-error target, so the result is accepted but not described as preferred precision.

## 2026-09-01 — Gate 6 evaluation route passed

- Evaluation uses each complete `520×696` held-out source image. Sensor images are resampled once to the fixed `0.5 µm/px` observer lattice (`671×898`), and target instances use nearest-neighbor resampling to that same lattice.
- The official frozen hard-label endpoint runs on MPS in bounded batches of four. No approximate or surrogate postprocessor is used for metrics.
- Each source/design pair contains seven deterministic planes (six headline off-focus depths plus the focus diagnostic) and 16 keyed Poisson endpoint conditions. Across 41 decontaminated test sources and five designs, the locked matrix contains 4,715 observer images.
- Three complete validation sources took `11.00–12.38 s` per source/design pair after native-grid resampling and frozen border filtering. The conservative maximum projects `2,536.9 s` (`42.3 min`) for the complete test matrix.
- A 25% contingency requires `3,171.1 s` (`52.9 min`), leaving `428.9 s` beyond that requirement inside the reserved one-hour local test window. Gate 6 therefore passes before any test metrics are read.
- Reproducible evidence is stored locally at `artifacts/runs/gate6/evaluation-route-benchmark.json`.

## 2026-09-01 — Global observer transform frozen

- The calibration pool is all 300 decontaminated training patches across the seven clear-pupil depths. A streaming 65,536-bin histogram gives global sensor percentiles `0.00155642` and `0.90136568`; labels and validation data do not influence these bounds.
- The resulting global affine transform is `clip((sensor - 0.0015564202) / 0.8998092622, 0, 1)`. Its clear-pupil in-focus validation PQ is `0.5952869`, compared with `0.5896305` for InstanSeg's standard per-image 0.1/99.9 preprocessing.
- The affine mapping passes the frozen requirement of no more than `0.03` absolute focus-PQ loss, so the predeclared asinh fallback is not activated. For context only, asinh also passed, with focus PQ `0.5948514`.
- Per-image normalization remains forbidden. The same frozen affine and clipping apply to every design, source, depth, and photon condition.
- Evidence is stored locally at `artifacts/runs/gate7/observer-transform-selection.json`.

## 2026-09-01 — Cubic baseline strength frozen

- The cubic control is the approved physical pupil phase `α(x³+y³)`, normalized so `α` is pupil RMS phase in radians. It uses the same Chromatix propagation, exposure, support, observer transform, and validation metric rules as every other design.
- Coarse strengths `{0.0, 0.6, 1.2, 1.8, 2.4}` were evaluated on the first 10 fixed validation sources. The best coarse strength was `0.0`, so `{0.0, 0.3, 0.6}` were evaluated on all 41 validation sources.
- The frozen mean off-focus PQ values were `0.5635504`, `0.5624525`, and `0.5606883`, respectively. The selected cubic RMS is therefore `0.0 rad`; under the predeclared validation rule, the tuned cubic baseline coincides with clear.
- This is retained as a valid negative baseline result rather than forcing a nonzero mask. The selected support-energy minimum is `0.9972982`, above the required `0.995`.
- Evidence is stored locally at `artifacts/runs/gate7/cubic-strength-selection.json`.

## 2026-09-01 — Surrogate scale and Adam settings frozen

- Thirty-two fixed training batches calibrated the approved proxy backward. The median sensor-gradient norm ratio is `κ = 108.9764975`; the surrogate uses this scale for every later step and never consults the exact VJP again.
- Exact-VJP Adam pilots used the same zero start and first 24 schedule batches at learning rates `0.01`, `0.03`, and `0.1`. Their 10-source validation mean off-focus PQ values were `0.5401327`, `0.5400166`, and `0.5329532`; worst-depth PQ values were `0.5238806`, `0.5194690`, and `0.5221003`.
- Learning rate `0.01` is frozen for exact-task, surrogate-backward, and image-fidelity runs. Adam uses `β1=0.9`, `β2=0.999`, and `ε=1e-8` for 120 matched steps from the shared zero start.
- Evidence is stored locally at `artifacts/runs/gate7/optimizer-calibration.json`.

## 2026-09-01 — Gate 8 matched one-start designs passed

- Exact-task, frozen-surrogate, and image-fidelity designs each ran 120 Adam steps from the shared zero start on the identical two-patch, three-depth schedule at learning rate `0.01`.
- Exact-task finished at `1.03297 rad` RMS and validation mean off-focus PQ `0.5795883`, an absolute `+0.0160379` over clear. Its validation focus PQ is `0.5887209`.
- The forward-identical surrogate finished at `0.14792 rad` RMS and off-focus PQ `0.5638640`. Image fidelity finished at `0.06908 rad` RMS and off-focus PQ `0.5642717`.
- Minimum large-reference support-energy fractions were `0.9972854`, `0.9972973`, and `0.9972978`, respectively, so the 96-pixel support remains valid.
- The exact design passes the pre-test one-start viability rule by improving validation off-focus PQ. This does not guarantee the much stronger frozen headline thresholds on the untouched test split.
- Evidence and complete traces are stored locally under `artifacts/runs/gate8/`.

## 2026-09-01 — Gate 9 locked test completed with a negative headline

- The pre-test freeze verifier passed before the first held-out metric was read. The
  evaluation then completed all 41 decontaminated test sources, five design rows, and
  4,715 official observer images in 2,104.2 seconds.
- Exact-task mean off-focus PQ is `0.6097963`, versus `0.5872808` for clear. The paired
  source-grouped difference is `+0.0225155`, with 95% bootstrap interval
  `[+0.0151338, +0.0302096]`.
- Exact task also exceeds cubic by `+0.0225155`, image fidelity by `+0.0223370`, and the
  forward-identical frozen-surrogate design by `+0.0200916`; every paired 95% interval
  has a positive lower bound.
- The worst-depth gain over clear is `+0.0405288`, and focus PQ improves by `+0.0135235`.
- The full frozen positive headline is rejected. Exact-task misses the required `+0.05`
  clear gain, worsens absolute count error by `2.53%` instead of reducing it by `10%`,
  and has non-positive mean differences at both Poisson endpoints.
- One official InstanSeg hard-postprocessing call emitted its upstream
  maximum-iteration warning. It returned normally, all expected rows were written, and
  no test rerun, threshold change, or post-hoc design change was made.
- Machine-readable evidence remains under `artifacts/runs/gate9/`; the readable audit,
  figures, limitations, and reproduction guide are under `outputs/`.
