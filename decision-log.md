# TessScope decision log

## 2026-09-02 — Source-derived LinkedIn fast cut passes final QA

- The selected candidate is `outputs/video/linkedin/tessscope-linkedin-fastcut.mp4`: exactly
  29.000 seconds, `1920×1080`, 30 fps, 870 frames, H.264/yuv420p, fast-start, no audio,
  3,656,025 bytes, and SHA-256
  `f511ec5e4609d3bd47b349aa2b2cf08b814df422511047ee6d4994e5d8635ee8`.
- Its only visual transform is uniform PTS remapping by `29/210`, producing exact playback
  speed `210/29`. There are no spatial filters, overlays, selective omissions, scene
  reordering, redrawn elements, or new animation.
- Nine representative output frames were compared with frames at their mapped source times
  using `t_source = t_output × 210/29`. Every comparison passed with correlation above
  `0.99995` and normalized mean absolute error below `0.00085`.
- Full decode, zero-black-interval, exact frame-count, codec, pixel-format, resolution,
  duration, and fast-start checks passed. A second full render was byte-identical.
- The source remains unchanged at SHA-256
  `767f5d03ae859d68b4d06b94a1398beda1a38026e1549bdb7b04606254a275a6`.
  The rejected custom `1080×1350` teaser is retained and remains unselected. No push,
  upload, publication, deployment, post, or submission was performed.

## 2026-09-02 — Custom 4:5 teaser rejected; source-derived fast cut selected

- The user rejected the newly designed `1080×1350` TessScope teaser as the LinkedIn
  candidate. It remains in the repository for history but is explicitly unselected; no
  file is deleted.
- The selected replacement must derive only from `outputs/video/tessscope-demo.mp4` and
  preserve every source frame's landscape composition, typography, figure layout, captions,
  color, ordering, and scientific content. No redesign, crop, reflow, overlay, selective
  omission, or reordered scene is allowed.
- Compress the complete 210.0-second source uniformly to exactly 29.0 seconds using the
  rational speed factor `210/29` (`7.241379310344827…×`). The only permitted content change
  is temporal; the intentionally silent audio stream may be omitted.
- Save the new candidate separately as
  `outputs/video/linkedin/tessscope-linkedin-fastcut.mp4`. No push, upload, publication,
  deployment, post, or submission is authorized.

## 2026-09-02 — LinkedIn teaser passes final production QA

- The final feed edit is exactly 27.000 seconds at `1080×1350`, 30 fps, H.264/yuv420p,
  fast-start, and 3.64 MB. It intentionally contains no audio stream because every
  essential word is burned into the muted-first visual.
- Three render iterations corrected unreadable scene pills, colliding optical captions,
  crowded result provenance, ghosted text transitions, and long visual holds. The final
  motion audit finds zero black intervals, zero decode errors, and zero frozen intervals
  longer than one second.
- Twenty-five encoded frames were inspected across scenes and transitions, including a
  nine-frame `320×400` mobile-preview contact sheet. The poster, hook, before/corrected
  wipe, three-runtime Tesseract boundary, pupil/PSF sweep, two separate result claims,
  and looped end lock-up remain readable and scientifically accurate.
- The teaser SHA-256 is
  `fb0d1bfa9459d12b80b26f49f52cf960e23029758c77df6fda650a4f149e296a`.
  A second full 810-frame render matched byte-for-byte. Source and claim hashes pass,
  `test_accessed=false`, and the original 3:30 judge video remains unchanged at its
  frozen SHA-256.
- No upload, publication, post, deployment, or submission was performed.

## 2026-09-02 — Feed-native LinkedIn teaser authorized

- Produce a separate 27-second `1080×1350` muted-first MP4 for LinkedIn. Preserve the
  existing 3:30 judge video and every frozen scientific artifact unchanged.
- Build the edit deterministically with the existing Pillow/FFmpeg toolchain. Scientific
  imagery must come only from traced TessScope microscopy, pupil, PSF, depth, architecture,
  and comparison assets; decorative light and typography may frame but never alter evidence.
- Keep the two positive frozen-validation claims separate: stage correction changes hard
  off-focus PQ from `0.4922` to `0.5523` with `74.7%` of hard frames improving; exact
  feedback gradients beat the forward-identical stopped-gradient control by `+0.0063` PQ
  with its 95% confidence interval above zero.
- The teaser may omit failed promotion gates and broader limitations but may not imply
  hardware validation, deployment, live inference, hidden-test performance, state of the
  art, or universal baseline superiority. No upload, publication, post, or submission is
  authorized.

## 2026-09-02 — Local release candidate passes every approved gate

- A Git archive of package milestone `b655405` passed the dependency-free demo check and
  served the judge page, JavaScript, site manifest, and figure manifest over loopback.
  It contained no Git metadata, active virtual environment, external data, or external
  model cache and required no download, Docker, retraining, or live inference.
- The final audit decision is GO: the dependency lock and Ruff pass, all 209 tests pass,
  the current tree and 611 history blobs have no release blockers, and the largest file
  is the 12.4 MiB traced validation replay.
- The four-page brief and 210-second 1080p30 video passed technical and visual review.
  A second complete video build matched byte-for-byte; the source-date-stabilized PDF is
  also byte-repeatable. The evidence is recorded in `outputs/release/GO-NO-GO.md`.
- This status is local only. No tag, remote, push, publication, deployment, upload, post,
  submission, locked-test access, or physical-microscope claim is authorized or performed.

## 2026-09-02 — Judge package assembled for local release verification

- The root README is now Track-05-first: the two-minute cached replay, protected causal
  claim, exact piecewise limitation, load-bearing Tesseract explanation, protocol, and
  scope appear before implementation detail. The full v1–v2.6 record moved to
  `docs/RESEARCH_HISTORY.md` without removing negative results.
- The editable four-page technical brief compiled to
  `output/pdf/tessscope-technical-brief.pdf`. Every page was rasterized and visually
  inspected; the final source-date-stabilized build is byte-repeatable.
- The 210-second 1080p30 H.264/AAC demo video uses only traced TessScope figures and the
  cached validation depth replay. Captions are baked in; the soundtrack is silent; the
  storyboard, narration, shot list, SRT, ffprobe metadata, source hashes, and output hash
  are preserved under `outputs/video/`.
- `docs/FULL_REPRODUCTION.md` separates the standard-library judge replay from the full
  Python 3.12 scientific path. New verified fetch helpers and isolated output options
  prevent accidental mutation of frozen artifacts and keep the BBBC006 test out of the
  public reproduction command.
- Repository-wide Ruff and all 209 tests pass. The subsequent clean-export verification
  and final release audit are recorded in the decision above.

## 2026-09-02 — Evidence-first visual and judge-demo phase completed

- The representative example was frozen before rendering as validation field `n21_s1`,
  crop origin `(0, 220)`, with `−2 µm` as its primary still. Selection minimizes robust
  distance from the 27-field population median across six preregistered evidence features.
- A deterministic 12.4 MiB replay cache contains the real JAX/Chromatix sensor outputs,
  frozen autofocus actions, official InstanSeg labels, physical pupil phases, and PSFs.
  Every recomputed PQ and residual depth matches its frozen source row.
- Five publication figures were generated in 300-DPI PNG, SVG, and PDF, plus seven depth
  frames, GIF, and H.264 MP4. Fixed display/crop rules, captions, alt text, and output
  hashes are recorded in the traceability chain.
- The local judge experience starts with `python scripts/serve_demo.py` and explicitly
  identifies itself as cached validation replay. It keeps the supported exact-versus-
  stopped result beside the non-significant piecewise comparison and all claim limits.
- Full regeneration is byte-stable. Repository-wide Ruff and all 208 tests pass. The
  BBBC006 test remains sealed; no push, publication, deployment, or physical-microscope
  claim was made.

## 2026-09-02 — Evidence-first visual and judge-demo phase authorized

- The user approved a new presentation phase after the terminal v2.6 result. It preserves
  all frozen scientific work and does not authorize new broad pupil optimization.
- The central claim is causal and validation-scoped: exact gradients across JAX optics,
  SciPy autofocus, and PyTorch InstanSeg improve corrected PQ over a forward-identical
  stopped-gradient system. V2.4 balanced step 14 is the primary evidence.
- The demo must show the significant exact-versus-stopped gain and the non-significant
  exact-versus-piecewise result together. PQ is a segmentation metric, not ordinary
  percent accuracy. V2.5/v2.6 negative follow-ups, sealed BBBC006 test, and lack of
  physical microscope validation remain explicit.
- Only training/validation evidence is allowed. Representative examples require a frozen
  quantitative selection rule; visual rendering may not influence selection.
- The deliverable is local-only: no push, publication, deployment, deletion, generated
  microscopy, or locked-test access is authorized.

## 2026-09-02 — V2.6 diagnosis-first phase authorized and preregistered

- V1–v2.5 remain immutable at local commit `09b9eb2`. V2.6 is a separately named
  improvement phase; no push, publication, deployment, submission, deletion, or test
  access is authorized outside its frozen gates.
- The primary objective is a corrected-PQ effect of at least `+0.005` over the strongest
  equally calibrated piecewise baseline, with a positive paired well-bootstrap lower
  bound, at least 60% positive wells, and positive leave-one-well-out means. Focus and
  frame-reliability targets are secondary and cannot buy a PQ regression.
- Before a new pupil/controller is generated, freeze and run an oracle/PQ-depth/controller
  decomposition plus fixed-total-photon exposure audit. The current balanced checkpoint,
  piecewise-028, source hashes, training-only partitions, controller family, exposure
  grid, decision rule, and test seal are preregistered in `configs/v2_6/`.
- Use B7 for the primary route unless the diagnostic supplies concrete evidence for more
  modes. Do not repeat v2.5's infeasible pupil-only pressure. Any exact optimization must
  restore feasibility before corrected-task optimization and must pass independent and
  full-loop derivative checks.
- If controller/exposure headroom is insufficient, freeze that result before considering
  a sequential two-mask route. Such a route requires primary/official evidence that rapid
  SLM switching is physically plausible and a fair matched two-mask piecewise baseline.

## 2026-09-02 — V2.6 oracle rejects controller-only rescue

- The balanced candidate's exact same-field zero-residual oracle reaches corrected hard
  PQ `0.554982`, below the minimum `0.556010` implied by piecewise-028 `+0.005`. Its
  oracle advantage over the actual piecewise correction is `+0.003972`, but the paired
  interval `[-0.001712, +0.009673]` includes zero and only 75.9% of frames improve.
- The complete 125-point monotone gain/bias/cubic grid was applied equally to balanced
  and piecewise depth estimates using same-field frozen PQ-depth interpolation. The best
  matched advantage is only `+0.001007`, interval `[-0.002973, +0.005238]`, with 51.9%
  positive wells and one negative leave-one-well-out mean.
- The controller shows low saturation (`0.74%`) but strong under-gain (`0.691` fitted
  slope), especially at `±6 µm`. Calibration reduces estimated focus error, but the pupil
  and observer ceiling prevents the required corrected-PQ effect.
- Decision before the exposure audit: do not launch joint B7 pupil/controller optimization
  from this diagnostic. Complete the fixed-total-photon audit, then apply the frozen route
  decision. The locked test remains sealed.

- Exposure smoke correction: the first seven development conditions were interrupted
  before artifact generation because a source hash was nested under `systems` and counted
  as a third design. The schema now requires exactly `candidate` and `baseline`; budgets,
  data, fractions, seeds, metrics, and gates are unchanged. No smoke metric is used.

## 2026-09-02 — V2.6 exposure audit rejects the primary route

- The fixed 400-photon audit evaluated balanced and piecewise-028 equally across five
  first/second allocations, three matched noise seeds, four development wells, and four
  untouched confirmation wells from the frozen training-only partitions: 1,440 paired
  nonzero-depth rows. No validation or test well entered selection.
- Both systems selected a `0.65/0.35` first/second allocation on development. On
  confirmation, balanced corrected PQ was `0.482311` versus piecewise `0.488784`, a
  difference of `-0.006473` with interval `[-0.041989, +0.017630]`. Only 50% of wells
  favored balanced and the minimum leave-one-out mean was `-0.016580`.
- Balanced confirmation focus MAE was `2.9833 µm`, signed direction was `55.56%`, and
  `59.72%` of frames improved. The first-frame preservation check passed, but all target
  performance, uncertainty, stability, focus, and prior-oracle checks failed.
- Decision: do not spend compute on the bounded pupil/controller/exposure route. Freeze
  it as a diagnosis-backed negative result and proceed only to primary/official research
  on whether sequential SLM masks justify a separately preregistered two-mask TessScope
  experiment. The locked test remains sealed.

## 2026-09-02 — Sequential SLM masks are physically defensible with qualifications

- A primary Optics Express experiment used a pupil-plane SLM to sequentially display
  multiplexed coded apertures while capturing real-space fluorescence images. Independent
  structured-illumination microscopy systems likewise advance SLM patterns between
  camera exposures using hardware triggers, including live-cell fluorescence work.
- Official Hamamatsu X15213 specifications give 60 Hz input, 10 ms rise, and 25 ms fall
  time at 532 nm. Meadowlark's official high-speed 1024×1024 LCoS specification gives
  roughly `<=1 ms` response at 532 nm, preloaded automated sequences, and hardware
  trigger boundaries. Two sequential exposures are therefore realistic on suitable
  hardware, but not instantaneous and not universal across SLMs.
- Detection-path fluorescence PSF engineering with an SLM has been experimentally
  demonstrated. The literature also warns that polarization, diffraction efficiency,
  and SLM losses matter in photon-limited fluorescence, so v2.6 retains a matched fixed
  total-photon comparison and labels hardware performance as unvalidated.
- Decision: the conditional active-acquisition route is physically defensible as an
  in-silico, hardware-ready experiment. Preregister two separate B7 sensing/capture
  pupils, a matched two-pupil piecewise family, switching assumptions, exact gradients,
  and all gates before optimization. Do not claim physical validation.

## 2026-09-02 — V2.6 completes as a negative diagnosis and two-mask experiment

- The two-mask served graph passed all registered central-difference checks: joint
  relative error `0.003168`, sensing-only `0.005255`, and capture-only `0.001991`, with
  all cosine agreements above `0.99997`. Exact/stopped forwards were identical and the
  sensing contribution was entirely carried through the stage-action path.
- All 76 frozen sensing/capture baseline pairs were physically valid. Forty-two preserved
  the matched development first-frame limit, but zero reached the `0.055` normalized
  residual bound or `1.0 µm` residual MAE; the best MAE was `1.52584 µm`.
- Four registered feasibility restorations and all 72 exact primary steps completed.
  Zero restoration was feasible. The best development final segmentation loss was
  `1.09807`, but that endpoint had `2.01435 µm` residual MAE and missed the matched
  first-frame limit; the lowest endpoint MAE was `1.99761 µm` and also failed residual.
- Decision: freeze v2.6 as a terminal negative training-only result. Zero candidates may
  advance to confirmation, hard validation, stopped-stage, derivative-free, or locked
  test evaluation. The BBBC006 test remains sealed.

## 2026-09-02 — V2.4 completes expanded hard validation without promotion

- All three frozen exact checkpoints and their three matched stopped-stage pupils were
  evaluated on the unchanged 45 valid wells, 27 hard-density wells, official InstanSeg
  endpoint, one predicted correction, and 2,000 paired-well bootstrap replicates.
- Exact first-frame PQ values are `0.492192`, `0.493238`, and `0.500103`, each more than
  `+0.01` above clear with a positive paired lower bound. Direction accuracy is
  `98.89–99.26%`, focus MAE is `1.252–1.295 µm`, and every correction reduces residual
  defocus and increases hard PQ.
- Balanced step 14 has the best corrected PQ, `0.552295`. It causally beats its matched
  stopped pupil by `+0.006259`, interval `[+0.000670, +0.011547]`, but beats v2.2
  piecewise-028 by only `+0.001285`, interval `[-0.001719, +0.004800]`. Only `55.6%` of
  hard wells favor it, below the frozen 60% stability gate, and its first PQ drop from
  segmentation-only is `0.010295`, just outside the `0.01` limit.
- The action-heavy and first-heavy checkpoints also miss the `+0.005` corrected piecewise
  effect and positive paired lower bound. Neither produces a qualifying causal gain over
  its stopped control.
- Decision: no v2.4 checkpoint passes all hard gates. Do not run the conditional matched
  derivative-free control or locked test. V2.5 is not activated because its approved
  trigger was zero soft-eligible v2.4 checkpoints, while the audit found 90. Freeze v2.4
  honestly as a negative validation result.

## 2026-09-02 — V2.4 finds and freezes three early-stopped candidates

- The audit reconstructed all 270 source records without retraining and served 269 unique
  parameter hashes on the unchanged four-batch/12-well exact soft pipeline. All nine
  frozen step-30 endpoints reproduced with maximum absolute difference `8.48e-8`.
- Ninety checkpoints passed the unchanged `1.098270310640335` first-frame ceiling,
  improved residual MAE relative to their source start, and passed finite B7/support
  checks. Fifty-seven canonical eligible checkpoints were globally nondominated.
- The pre-registered hash deduplication, one-per-source-run diversity, and tie-break select
  balanced segmentation-only step 14, action-heavy segmentation-only step 12, and
  first-heavy segmentation-only step 9. Their first losses are `1.097063`, `1.095286`,
  and `1.093136`; residual MAEs are `1.174831`, `1.175803`, and `1.367156 µm`.
- All 270 pupils exceed the `0.995` support threshold; the observed minimum is `0.997226`.
  Soft audit time was `1,272.2 s`. Test and hard-label access are false.
- Decision: freeze the three exact checkpoints before running matched stopped-stage
  optimizations and expanded hard validation. Do not alter their steps or parameters.

## 2026-09-02 — V2.4 matched stopped-stage controls frozen

- Three stopped-stage runs used the identical segmentation-only start, objective profile,
  training batches, Adam schedule, and selected budgets of 14, 12, and 9 steps. Only the
  gradient through the autofocus stage action and its downstream terms was stopped.
- Exact and stopped modes have identical forward validation metrics with maximum absolute
  difference `0.0` at all three stopped endpoints.
- The stopped endpoints retain first-frame losses `1.090614`, `1.090522`, and `1.090406`,
  but residual MAEs are `2.083919`, `2.028220`, and `1.887137 µm`. Their paired exact
  candidates reach `1.174831`, `1.175803`, and `1.367156 µm`, respectively.
- Decision: the exact feedback gradient materially improves the soft correction outcome.
  Freeze these six pupils before applying the expanded hard protocol.

- Pre-hard stability definition: for first-frame PQ versus clear, corrected PQ versus
  v2.2 piecewise-028, and corrected PQ versus the matched stopped-stage pupil, at least
  60% of hard-density wells must have a positive within-well mean difference. The overall
  mean must also stay positive after leaving out each individual well. Because the frozen
  protocol uses one deterministic field per well, this explicitly rejects single-field
  dependence. This rule is recorded before any v2.4 hard metric is computed.
- The frozen selection manifest's contract hash was refreshed after adding this pre-hard
  stability clause. Its three checkpoint names, steps, parameters, hashes, soft metrics,
  and source evidence remain unchanged.

## 2026-09-02 — V2.4 frozen-checkpoint audit authorized

- V1 through v2.3 remain immutable; v2.3 is frozen at commit `93da502` as a correct
  negative endpoint-only experiment. V2.4 uses a separate namespace and source artifact
  `artifacts/runs/v2_3/optimization/closed-loop-matrix.json`.
- The v2.4 pool is exactly the 30 saved post-update phase vectors from each of nine exact
  v2.3 runs (270 checkpoint records). No retraining or altered vector may enter the first
  audit. The three unique starts are controls only and cannot be selected as optimized
  candidates.
- Eligibility retains the exact first-frame loss maximum `1.098270310640335`, requires
  residual MAE strictly below that run's frozen start, finite metrics, and a valid pupil.
  Evaluation uses the unchanged calibration, normalization, exposure, autofocus,
  observer, and 12 fixed validation wells.
- Eligible checkpoints are filtered by three-objective nondominance. The existing v2.3
  tie-break applies; no more than one checkpoint per source run can be selected, and at
  most three source runs advance. Exact parameter-hash duplicates are evaluated once but
  every source checkpoint remains recorded.
- Aggregate v2.3 endpoint results are known. No intermediate-checkpoint validation value
  has been used for selection before this preregistration. Hard labels cannot select
  among the 270 checkpoints.
- The integrity freeze found 269 unique parameter hashes among 270 source records. The
  only duplicate is step 1 of the balanced and first-heavy piecewise-028 runs; it will be
  served once and reported under both frozen checkpoint identifiers. Exact duplicates
  also cannot occupy two promotion slots; the lexicographically smallest checkpoint name
  is the canonical selection identity.
- If no checkpoint qualifies, preserve v2.4 and activate separately named v2.5
  constrained exact-gradient optimization. If checkpoints qualify, freeze them before
  stopped-stage and hard validation; derivative-free and locked-test work remain
  conditional on every unchanged v2.3 gate.

## 2026-09-02 — V2.3 closed-loop objective pre-registered

- V2.2 remains frozen at commit `cda7d34` with the locked test sealed. V2.3 uses a new
  namespace and does not alter the v2.2 piecewise frontier or any earlier result.
- One B7 pupil forms both exposures. The first optics call generates autofocus support
  and query frames; the frozen SciPy ridge model predicts a stage action clipped to
  `[-6, +6] µm`; residual depth is true depth plus that action; a second B7 optics call
  forms the corrected query frames for the same frozen InstanSeg observer.
- The primary exact profile weights final corrected-frame segmentation `1.0`, first-frame
  segmentation `0.35`, normalized squared residual depth `0.10`, normalized squared stage
  magnitude `0.01`, and the fixed second-exposure cost `0.01`. First-heavy and
  action-heavy profiles change only one predeclared weight. No profile is chosen after
  hard validation.
- Three frozen starts are allowed: B7 segmentation-only, the v2.1 joint pupil, and v2.2
  piecewise-028. Each profile/start receives the same 30-step Adam budget and validation
  schedule. Up to three soft-nondominated endpoints may advance to the 45-well hard gate.
- Exact and stopped-stage gradients are forward-identical. The exact full-loop directional
  derivative must retain relative error below `0.01` and cosine above `0.99`; the stage
  path must contribute at least 1% of total gradient norm in one predeclared probe.
- Promotion additionally requires inherited first-frame segmentation/focus gates,
  beneficial correction, at least `+0.005` corrected hard PQ beyond the matched v2.2
  piecewise correction with a positive paired-well lower bound, and at least `+0.005`
  corrected PQ over a forward-identical stopped-stage optimization with a positive lower
  bound. A matched derivative-free run is conditional on clearing those earlier gates.

## 2026-09-02 — Exact v2.3 feedback derivative passes and is load-bearing

- A separate v2.3 B7 optics Tesseract marks depth as differentiable. The first optics
  call and SciPy autofocus remain unchanged; the predicted clipped stage action now sets
  residual depths for a second optics call with the same pupil and an independent noise
  realization before the final frozen InstanSeg task loss.
- The complete served feedback objective passes five-direction central differences with
  overall median relative error `0.004802` and cosine `0.999967` across stable `1e-3` and
  `1e-4` epsilon windows. Test access is false.
- Exact and fully stopped-stage modes have identical forward value to machine precision.
  Their corrected gradient difference is `82.36%` of the exact gradient norm at the
  predeclared piecewise-028 probe, far above the 1% load-bearing threshold.
- Decision: the feedback connection is both numerically valid and materially active.
  Proceed to the matched profile/start optimization matrix; do not promote from this
  derivative result alone.

- Pre-optimization audit clarification: an initial partial-stop diagnostic reported
  `68.0%`. Stopped-stage mode now stops the action for both
  the second optics call and the residual/action penalties. This makes the ablation a
  complete stop of every downstream action path. Exact-mode values and derivatives are
  unchanged; the corrected forward-identical ablation evidence supersedes only the
  earlier partial-stop diagnostic in Git history.

## 2026-09-02 — V2.3 completes the exact matrix with no soft promotion

- All nine pre-registered combinations of three objective profiles and three frozen
  starts completed 30 exact-gradient Adam steps each, for 270 total steps. The fixed
  training and 12-well validation schedules were unchanged, and test access is false.
- Every endpoint reduced validation residual-defocus MAE relative to its start. The
  lowest residual was `0.934414 µm`, down from `1.325543 µm`, for the action-heavy
  projected-joint run; its first-frame segmentation loss was `1.116030`.
- The closest promotion result was first-heavy piecewise-028. It reduced residual MAE
  from `1.191849` to `0.999509 µm` and reached final-frame segmentation loss `1.061714`,
  but first-frame loss `1.100370` exceeded the frozen `1.098270` ceiling by `0.002099`.
- Decision: do not relax the ceiling or select an intermediate checkpoint. With zero
  soft-eligible exact endpoints, the contract does not authorize stopped-stage
  optimization, expanded hard labels, the matched derivative-free control, or the
  locked test. Freeze v2.3 as a complete negative validation experiment.


## 2026-09-01 — V2.2 matched piecewise-frontier protocol pre-registered

- V2.1 remains frozen at commit `4a88bdd` as a negative pre-test result. V2.2 does not
  tune or relabel `b7-projected-0.50-step-30`; its coefficients and source hashes are
  frozen in `configs/v2_2/frozen-joint-candidate.json` before new frontier results.
- The baseline family uses only the matched B7 segmentation and focus pupils: convex
  interpolation at `0.05` spacing, focus injection through `1.50` at `0.05` spacing,
  and nonnegative two-weight mixtures from `0` through `1.50` at `0.25` spacing. The
  original naive sum is included exactly. Only mixtures outside the physical ball are
  radially projected to `2.4975` radians; interior amplitudes remain interior.
- All unique mixtures receive the unchanged 12-well served soft evaluation. Hard labels
  remain unused until soft screening freezes the non-dominated envelope, near-envelope
  guard band, family anchors, and operating-point brackets. Every retained point then
  receives the same 45-well/27-dense-well hard endpoint and paired-well uncertainty.
- A v2.2 frontier effect is nontrivial if the joint pupil gains at least `0.10 µm` focus
  MAE at equal-or-better hard PQ, or at least `0.005` hard PQ at equal-or-better MAE.
  `0.10 µm` is 5% of the 2-µm acquisition-plane spacing; `0.005` is half the frozen
  segmentation-tolerance width. The relevant paired 95% bootstrap lower bound must also
  be positive, so rounding-scale differences cannot promote a candidate.
- Hypervolume is secondary and uses a frozen reference `(0, 0)` after normalizing PQ
  between clear and B7 segmentation-only and focus benefit between B7 segmentation-only
  and the original naive sum. It cannot override a failed matched operating-point gate.
- Every prior clear-gain, segmentation-retention, focus, stage, derivative, gradient-free,
  support, photon, well-grouping, and test-hygiene rule remains unchanged. If v2.2 fails,
  preserve it and proceed to the separately approved v2.3 closed-loop feedback experiment.

## 2026-09-01 — V2.2 soft screen freezes 38 hard-frontier points

- The 101 predeclared family aliases collapse to 87 unique physical coefficient vectors.
  Fourteen require radial projection; all interior amplitudes remain unchanged. The
  original naive sum reproduces validation segmentation/focus values
  `1.12248264/1.45912194`, and the segmentation endpoint reproduces
  `1.09027031/4.59834081` through the unchanged served chain.
- The predeclared soft rule retains 38 points: every nondominated point, the fixed guard
  band, family anchors, the original sum, and operating-point brackets. No hard instance
  label or v2.2 hard metric was used for this selection.
- Every retained point remains inside the 2.5-radian RMS ball and exceeds the unchanged
  `0.995` minimum PSF support-energy fraction across all seven depths. Quantization
  diagnostics are recorded for each retained point.
- Decision: freeze these 38 identifiers for the expanded hard run. Do not narrow them
  after seeing hard outcomes; the deliberately generous baseline budget is part of the
  credibility of the frontier comparison.

## 2026-09-01 — V2.2 fails the fair frontier and activates v2.3

- All 38 frozen mixtures were evaluated on 45 valid wells and seven depths. The artifact
  contains 11,970 primary rows and 6,156 hard-density correction rows; all three inherited
  well exclusions and the 27-well hard subset are unchanged. Test access is false.
- `v2_2-piecewise-028`, a focus-injection mixture, reaches hard PQ `0.50254290`, direction
  `100%`, and focus MAE `1.20717865 µm`. It slightly improves both primary coordinates
  over the frozen joint pupil's `0.49606437` PQ and `1.21073477 µm` MAE.
- At the matched segmentation operating point, piecewise-minus-joint focus advantage is
  `-0.00356 µm` with paired 95% interval `[-0.07681, +0.06873]`; the joint neither reaches
  the `0.10 µm` effect nor a positive lower bound. At matched focus, joint-minus-piecewise
  PQ is `-0.00648` with interval `[-0.01277, -0.00068]`, statistically favoring the
  piecewise point rather than the joint.
- The normalized piecewise hypervolume is `0.90708`; adding the frozen joint contributes
  exactly zero. A separate two-weight point also improves both PQ and MAE over the
  original naive sum, confirming that the earlier single-point baseline was incomplete.
- Decision: v2.2 is a negative validation result. Do not run a conditional matched B7
  derivative-free search or the locked test because the earlier frontier prerequisite
  already failed. Freeze hashes, preserve every row, and proceed to the separately named,
  approved v2.3 closed-loop differentiable microscope.


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

## 2026-09-01 — B11 completes the basis ladder with a negative pre-test result

- Matched 90-step B11 baselines used the same 36 training and 12 validation wells.
  Segmentation-only reached validation task loss/focus MSE `1.09020/4.31924`;
  focus-only reached `1.16014/2.81038`; constrained naive superposition reached
  `1.15152/1.38904`.
- Segmentation-primary projected runs at balances `1`, `2`, and `3` used matched 40-step
  schedules. Preselected balance `2` step 15 and balance `3` step 15 satisfied the
  unchanged training and soft-validation segmentation allowances before expanded hard
  labels were evaluated.
- Expanded validation used the same maximum 45/48 valid wells, 27-well hard-density
  subset, seven depths, deterministic fields, correction frames, and 2,000 paired
  well-bootstrap replicates as B7. No validation exclusions or definitions changed.
- Balance `2` reaches hard PQ `0.49157`, a `+0.03545` gain over clear with 95% interval
  `[+0.00996, +0.05966]`, direction `100%`, and focus MAE `1.06676 µm`. Its `0.01206`
  drop from B11 segmentation-only exceeds the allowed `0.01`, and it does not beat B11
  naive superposition's `0.80987 µm` focus MAE.
- Balance `3` reaches hard PQ `0.49044`, a `+0.03433` gain over clear with 95% interval
  `[+0.00967, +0.05699]`, direction `100%`, and focus MAE `1.05502 µm`. Its `0.01318`
  segmentation drop and superposition comparison fail the same gates.
- The matched B11 derivative-free search was explicitly conditional on first clearing
  the segmentation and naive-superposition gates. Neither candidate did, so that run
  could not affect eligibility and was not performed.
- Decision: stop v2.1 at a complete negative pre-test checkpoint. Freeze the result and
  hashes in `configs/v2_1/pretest-block.json`; do not access the BBBC006 locked test,
  generate promotion-only claims, or reinterpret thresholds. A future attempt requires
  a separately approved hypothesis rather than more tuning of this basis ladder.

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

## 2026-09-02 — V2.5 hard-near-miss experiment authorized and preregistered

- The user explicitly authorized a new v2.5 experiment after v2.4's best exact checkpoint
  missed the B7 hard segmentation tolerance by only `0.0002953` PQ and showed a causal
  `+0.006259` corrected-PQ gain over its matched stopped-stage control.
- The new hypothesis is separate from the older v2.4 zero-soft-eligible fallback, whose
  inactive status remains historically correct.
- B7 is frozen as a continuity control. B11 Noll modes 5–15 are primary under the same
  open `2.5 rad` RMS ball, support, calibration, controller, observer, and hard gates.
- The primary method is an exact-gradient augmented Lagrangian with explicit first-frame
  and residual constraints, not a fixed weighted objective. A bounded exact-Jacobian
  SLSQP alternate is registered before results.
- Source hashes, exact B7→B11 zero-padding, five B11 starts, two B7 starts, epsilon ladder,
  budgets, strong B11 piecewise family, derivative requirements, and one-run test seal are
  frozen in `configs/v2_5/` before any new pupil is generated.

## 2026-09-02 — V2.5 B11 full-loop derivative passed

- The new differentiable-depth B11 optics service was checked at the frozen B11
  segmentation-only start before constrained optimization.
- The full served B11 optics → SciPy autofocus → InstanSeg → stage action → residual-depth
  second exposure derivative has median relative error `0.0090221` and cosine agreement
  `0.9998874`, passing the unchanged `<0.01` and `>0.99` gates.
- The exact-minus-stopped stage-path gradient is `43.06%` of the full gradient norm and
  exact/stopped forward values are identical, so the controller path is load-bearing.
- The BBBC006 locked test was not accessed.

## 2026-09-02 — Invalid minibatch-constraint smoke stopped

- The first eight B11 optimizer smoke steps applied the registered mean first-frame
  constraint to one noisy batch at a time. Per-batch task losses vary far more than the
  `0.008` margin, so scaled first-constraint violations alternated between about `-23`
  and `+15`; those dual updates did not represent the registered mean constraint.
- The run was interrupted, marked non-scientific, and preserved in
  `artifacts/runs/v2_5/diagnostics/invalid-minibatch-smoke.json`.
- Valid v2.5 primary steps differentiate one augmented objective formed from the exact
  mean of all four frozen training batches. No threshold, step count, epsilon, start, or
  promotion rule changed, and the locked test remains sealed.

## 2026-09-02 — Invalid SLSQP anchor smoke stopped

- After all 15 B11 primary stages completed without a feasible endpoint, the registered
  two-batch SLSQP alternate activated. Its first three exact evaluations revealed that
  the two-batch constraint values were being compared with the four-batch primary anchor,
  which made first-frame feasibility artificially loose.
- Those evaluations were interrupted and marked non-scientific in
  `artifacts/runs/v2_5/diagnostics/invalid-slsqp-anchor-smoke.json`.
- Valid alternate runs use the B11 segmentation-only mean on the identical two frozen
  constraint batches plus the unchanged `0.008` margin. The two-batch budget, starts,
  residual bounds, hard thresholds, and test seal are unchanged.

## 2026-09-02 — V2.5 B11 constrained matrix completed with zero promotions

- All five frozen B11 starts completed the three registered residual bounds with 18 exact
  aggregate augmented-Lagrangian steps per stage: 15 primary endpoints and 270 steps.
- No primary endpoint passed the frozen training/soft feasibility rules, so the registered
  exact-Jacobian SLSQP alternate activated for all three residual bounds.
- Every SLSQP run reached the eight-iteration limit without satisfying both matched
  training constraints. Their final scaled `(first, residual)` violations were
  `(0.979, 1.612)`, `(1.699, 1.234)`, and `(2.168, 1.176)`.
- The 12-well validation selector found zero eligible B11 endpoints and froze zero
  promotions. Therefore B11 piecewise, matched stopped-stage, hard-label,
  derivative-free, controller-gain, and locked-test work are not authorized by the
  registered ordering. The required B7 continuity matrix remains to be completed.

## 2026-09-02 — V2.5 completes as a negative constrained experiment

- Both B7 frozen starts completed all three residual bounds with 18 exact aggregate
  augmented-Lagrangian steps per stage: six primary endpoints and 108 steps. None passed
  the matched training constraints and unchanged 12-well soft ceiling.
- The registered B7 SLSQP fallback completed three eight-iteration runs using 53 exact
  vector evaluations. Their final scaled `(first, residual)` violations were
  `(1.000, 2.582)`, `(1.253, 2.284)`, and `(1.803, 1.739)`.
- Across B11 and B7, v2.5 completed 21 primary endpoints, 378 exact aggregate steps, six
  SLSQP runs, and 105 exact fallback evaluations. Zero B11 candidates were selected.
- Decision: freeze v2.5 as a complete negative training/validation experiment. Do not
  generate the conditional B11 piecewise family or run stopped-stage, new hard-label,
  derivative-free, gain, or locked-test work. The test remains sealed and test access is
  false in every frozen v2.5 artifact.

## 2026-09-02 — Judge demo foregrounds frozen evidence and Tesseract boundaries

- The first viewport now uses a deterministic four-panel visual from the already frozen
  `n21_s1`, −2 µm validation replay. It adds no sample selection, normalization, inference,
  or scientific claim, and it explicitly says the frame is not the population estimate.
- The architecture figure now labels the optics, autofocus, and observer Tesseract APIs,
  including the two calls to the same optics service and the exact VJP across all runtime
  boundaries. This matches `src/tessscope/v2_3/closed_loop.py`.
- The default replay keeps the preregistered −2 µm field and immediately discloses that its
  exact corrected PQ (`0.630`) is slightly below stopped (`0.631`); the supported causal
  result remains the paired comparison across 27 validation wells.
- Mobile navigation links use a 44 px minimum height. The cached/local mode, limitations,
  locked-test seal, and no-physical-microscope boundary remain unchanged.

## 2026-09-02 — Local Track 05 release-candidate packaging authorized

- The official 2026 challenge page confirms the required composition, end-to-end-gradient,
  Apache-2.0, public-repository, reproducible README, named-track write-up, and optional
  five-minute-video criteria. TessScope will enter “Track 05 — Differentiable graphics &
  rendering” because its learned phase pupil differentiates through wave-optical rendering,
  autofocus, and a frozen segmentation observer.
- The public claim remains the frozen v2.4 exact-versus-stopped causal result. The
  non-significant piecewise comparison, negative v2.5/v2.6 follow-ups, validation-only
  scope, sealed BBBC006 test, and absent physical-microscope validation remain visible.
- Project-authored code and documentation use the unmodified Apache License 2.0. BBBC
  data, InstanSeg models, dependency code, embedded font glyphs, and trademarks keep their
  own terms; no downloaded dataset archive or pretrained weight is part of the candidate.
- The initial GitHub inventory found no current or historical blob above 13,035,355 bytes,
  no raw dataset archive or model weight, no absolute private path, and no redacted secret
  pattern. The Git database is about 29 MiB and no remote is configured.
- Local packaging is authorized. Creating a remote, pushing, tagging, uploading video,
  publishing, posting, deploying, or submitting still requires a new explicit approval.
