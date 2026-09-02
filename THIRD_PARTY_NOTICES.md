# Third-party notices

TessScope code and original documentation are available under Apache License 2.0.
That license does not replace the terms of the software, data, models, fonts, or
trademarks listed below.

## Data and annotations represented in tracked outputs

| Component | Use in TessScope | Terms and attribution |
|---|---|---|
| BBBC006v1 out-of-focus U2OS images and automated CellProfiler labels | Source material for the compact validation replay, traced microscopy panels, and v2 figures | The [official BBBC006 page](https://bbbc.broadinstitute.org/BBBC006) states that copyright and related rights in the images and ground truth are waived to the extent possible under law. Cite Ljosa et al., *Nature Methods* 2012, and acknowledge Sigrun Gustafsdottir, Tom Hasaka, and Mark Bray as requested by the dataset page. |
| BBBC039v1 images and masks | Source material for the v1 evaluation figures | CC0, as recorded by the [official BBBC039 page](https://bbbc.broadinstitute.org/BBBC039) and [`data/manifests/bbbc-assets.json`](data/manifests/bbbc-assets.json). Cite Caicedo et al. 2018 and Ljosa et al., *Nature Methods* 2012. |
| BBBC038v1 stage-1 training images | Used only for identity decontamination review against the frozen observer's disclosed training data | CC0, as recorded by the [official BBBC038 page](https://bbbc.broadinstitute.org/BBBC038) and [`data/manifests/bbbc-assets.json`](data/manifests/bbbc-assets.json). |

The repository intentionally excludes the downloaded BBBC006, BBBC038, and BBBC039
archives. The tracked `artifacts/runs/demo/validation-replay.npz` is a 12.4 MiB derived
validation pack containing one preregistered BBBC006 field, simulated sensor frames,
automated reference labels, frozen model predictions, and traced optical arrays. It does
not contain the approximately 2.5 GB selective BBBC006 download.

## Model and scientific software

| Component | Version or pin | License | Distribution status |
|---|---|---|---|
| [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) | 1.12.0 | Apache-2.0 | Dependency only; not vendored. |
| [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) | 0.4.1 | Apache-2.0 | Dependency only; not vendored. |
| [Tesseract-Torch](https://github.com/pasteurlabs/tesseract-torch) | 0.1.0 | Apache-2.0 | Dependency only; not vendored. |
| [Chromatix](https://github.com/chromatix-team/chromatix) | commit `ce1482906fc663298613bb252bbf425e3be59839` (package 0.6.0) | MIT | Dependency only; not vendored. Cite Deb et al., “Chromatix,” bioRxiv 2025, DOI `10.1101/2025.04.29.651152`. |
| [JAX](https://github.com/jax-ml/jax) | resolved by `uv.lock` | Apache-2.0 | Dependency only; not vendored. |
| [NumPy](https://github.com/numpy/numpy) | resolved by `uv.lock` | BSD-3-Clause and bundled notices | Dependency only; not vendored. |
| [SciPy](https://github.com/scipy/scipy) | resolved by `uv.lock` | BSD-3-Clause and bundled notices | Dependency only; not vendored. |
| [PyTorch](https://github.com/pytorch/pytorch) | resolved by `uv.lock` | BSD-style plus bundled notices | Dependency only; not vendored. |
| [InstanSeg](https://github.com/instanseg/instanseg) | `instanseg-torch` 0.1.1 | Apache-2.0 | Dependency only; not vendored. Cite Goldsborough et al., arXiv 2024, DOI `10.48550/arXiv.2408.15954`. |
| InstanSeg `single_channel_nuclei` | model release 0.1.2 | Apache-2.0 according to the upstream model index and local manifest | Model weights are not committed. The official URL and SHA-256 are in [`data/manifests/instanseg-v0.1.2.json`](data/manifests/instanseg-v0.1.2.json). |
| [CellProfiler](https://github.com/CellProfiler/CellProfiler) | dataset annotation provenance only | BSD-3-Clause | No CellProfiler code is vendored. BBBC006 automated labels were produced with an upstream CellProfiler pipeline. |

Other direct Python dependencies are resolved in `uv.lock` and installed from their
upstream distributions. No dependency source tree or binary wheel is committed here.
Their upstream licenses continue to apply.

## Fonts and media tooling

- Matplotlib-generated SVG and PDF figures may embed or outline glyphs from DejaVu Sans.
  DejaVu fonts use the permissive [DejaVu Fonts license](https://dejavu-fonts.github.io/License.html),
  derived from the Bitstream Vera license. No standalone font file is committed.
- The demo MP4 was encoded locally with FFmpeg/libx264. No FFmpeg or x264 binary, library,
  or source code is distributed in this repository. The video contains no music or
  third-party voice recording.
- Product and project names remain trademarks of their respective owners. Reference to
  Tesseract, Chromatix, InstanSeg, CellProfiler, JAX, NumPy, SciPy, PyTorch, or BBBC is
  descriptive and does not imply endorsement.
