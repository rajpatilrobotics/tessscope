# TessScope polished end-to-end demo — exact on-screen copy

## Global

- `TessScope`
- `FROZEN VALIDATION`

## 0.0–3.0 s — Hook and input

- `TessScope · differentiable closed-loop microscopy`
- `Can one exact gradient teach a microscope to refocus?`
- `BBBC006 · U2OS Hoechst nuclei`
- `Real fluorescence input begins the differentiable loop`

## 3.0–7.0 s — Pupil and PSF hero

- `LEARNED OPTICS`
- `A phase pupil shapes focus across depth`
- `EXACT B7 PHASE PUPIL`
- `CHROMATIX PSF SWEEP`
- `−6 µm`, `−4 µm`, `−2 µm`, `0 µm`, `+2 µm`, `+4 µm`, `+6 µm`
- `Seven depths · one learned pupil`

## 7.0–12.0 s — Forward journey

- `FORWARD JOURNEY`
- `Three Tesseracts compose one workflow`
- `OPTICS TESSERACT`
- `JAX · Chromatix`
- `T1 first exposure · T2 corrected exposure`
- `AUTOFOCUS TESSERACT`
- `NumPy · SciPy`
- `focus estimate · stage action`
- `OBSERVER TESSERACT`
- `PyTorch · InstanSeg`
- `nuclei observation · objective J`
- `INPUT → T1 → AUTOFOCUS → STAGE → T2 → OBSERVER → J`
- `Three Tesseracts · three native runtimes · one differentiable workflow`

## 12.0–16.0 s — Reverse gradient

- `EXACT REVERSE GRADIENT`
- `One gradient crosses every boundary`
- `OBJECTIVE J → OBSERVER → OPTICS T2 → AUTOFOCUS ACTION → OPTICS T1`
- `θ5`, `θ6`, `θ7`, `θ8`, `θ9`, `θ10`, `θ11`
- `Seven phase parameters update end-to-end`

## 16.0–21.0 s — Correction

- `AUTOFOCUS CORRECTION`
- `See defocus → predict stage action → expose again`
- `FIRST EXPOSURE`
- `−2.86 µm focus estimate`
- `STAGE ACTION`
- `CORRECTED EXPOSURE`
- `Matched −6 µm field · identical display transform`

## 21.0–25.0 s — Correction outcome

- `CORRECTION OUTCOME`
- `PQ 0.4922 → 0.5523`
- `74.7% of hard frames improve`
- `27 hard-density wells`

## 25.0–27.0 s — Gradient evidence

- `CAUSAL GRADIENT EVIDENCE`
- `Exact vs forward-identical stopped gradient`
- `+0.0063 PQ`
- `95% CI [+0.0007, +0.0115]`
- `27 hard-density wells · 2,000 grouped bootstrap replicates`

## 27.0–29.0 s — Lockup

- `TessScope`
- `Differentiable closed-loop microscopy`
- `Tesseract Hackathon 2026 · Track 05`
- `Exact gradients across JAX · NumPy/SciPy · PyTorch`

## Claim boundary

PQ is reported only as panoptic quality on frozen validation evidence. The edit does not
claim hidden-test performance, physical-microscope validation, live inference, state of the
art, accuracy percentage, universal superiority, or superiority over every baseline.
