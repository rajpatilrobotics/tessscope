# TessScope video shot list and provenance

| Shot | Source file | Source role | Traceability |
|---|---|---|---|
| Evidence hero | `outputs/demo/figures/hero-evidence.png` | Frozen representative validation field, phase pupil and PSF | `outputs/demo/figure-manifest.json`; `outputs/demo/traceability-manifest.json` |
| Runtime architecture | `outputs/demo/figures/gradient-architecture.png` | Served forward path, exact VJP and stopped-action control | Derivative values traced to `artifacts/runs/v2_3/gates/closed-loop-derivative.json` |
| Pupil/PSF panel | `outputs/demo/figures/pupil-psf-depth.png` | Frozen phase parameters and simulated PSFs | Replay array hashes in the traceability manifest |
| Depth animation | `outputs/demo/replay/validation-depth-sweep.mp4` | Seven-plane cached validation replay | Per-frame hashes in `outputs/demo/figure-manifest.json` |
| Matched microscopy | `outputs/demo/figures/matched-microscopy.png` | Same field, transform, labels and three compared systems | `artifacts/runs/demo/validation-replay.json` |
| Population curves | `outputs/demo/figures/pq-focus-depth.png` | 27-well primary-density validation summaries | Source JSON pointers in the traceability manifest |
| Causal comparison | `outputs/demo/figures/causal-comparison.png` | Paired exact/stopped and exact/piecewise effects | `artifacts/runs/v2_4/validation/expanded-hard-validation.json` |

The rendered video adds only timing, scaling, a dark matte, and human-written captions.
It does not modify, synthesize, or interpolate scientific evidence.
