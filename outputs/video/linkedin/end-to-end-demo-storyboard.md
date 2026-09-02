# TessScope end-to-end demo storyboard

Status: approved production contract  
Format: 29.0 seconds, 1920×1080, 30 fps, muted-first  
Evidence boundary: frozen validation only; no locked-test or physical-microscope claim

| Time | Scene | Visual action | Purpose |
| --- | --- | --- | --- |
| 0.0–3.0 s | Problem and input | A traced BBBC006 U2OS Hoechst validation patch enters a clean scientific frame beside the project question. | Establish the real-data input and the autofocus problem without implying a live microscope. |
| 3.0–7.0 s | Pupil, PSF, first exposure | The exact learned phase pupil resolves into the seven-depth PSF family, then into the first simulated exposure. | Show the differentiable optics path in causal order. |
| 7.0–11.5 s | Forward product journey | A signal advances through Optics T1, Autofocus T, the predicted stage action, Optics T2, Observer T, and the objective. | Explain the three native runtimes as one composed workflow. |
| 11.5–15.5 s | Exact reverse gradient | The objective sends a highlighted reverse path through Observer, Optics T2, the autofocus action, and Optics T1 into seven phase parameters. | Make the end-to-end gradient path legible and distinct from the forward pass. |
| 15.5–20.5 s | Closed-loop correction | A matched −6 µm field progresses through defocus detection, a predicted focus offset, and a corrected exposure. | Show the strongest concrete closed-loop demo moment with identical display transforms. |
| 20.5–24.5 s | Descriptive validation result | The matched field remains visible while the aggregate stage-correction result is presented separately. | Report PQ 0.4922 → 0.5523 and 74.7% improvement without treating one image as the population. |
| 24.5–27.0 s | Causal gradient evidence | A single exact-versus-stopped confidence-interval plot draws on screen. | Isolate the forward-identical control and the +0.0063 PQ gradient-path contribution. |
| 27.0–29.0 s | TessScope lockup | The input-to-optimized-pupil journey resolves into the TessScope title and validation-only qualifier. | Give the title enough time to register at small-player size. |

## Visual contract

- Warm white canvas, deep navy typography, and restrained cyan, magenta, and orange scientific accents.
- Fixed title, content, explanation, and evidence bands; no text is placed over microscopy.
- Symmetrical grids, generous whitespace, and minimum 36 px supporting copy at delivery resolution.
- Scientific arrays keep their aspect ratio and frozen display transforms. No per-image normalization, sharpening, denoising, or fabricated imagery is permitted.
- Reverse-gradient motion uses the same nodes as the forward pass so the direction change is unambiguous.
- Descriptive stage-correction evidence and causal exact-versus-stopped evidence remain separate scenes.

## Review checkpoints

- Encoded-frame inspection at all eight scenes and both sides of each transition.
- Separate 640×360-per-tile contact sheet for text collision and small-player readability.
- Full decode, exact frame-count/duration, codec/pixel-format, fast-start, black-frame, and long-freeze checks.
- Source hashes and displayed values checked against the frozen replay and validation JSON.

