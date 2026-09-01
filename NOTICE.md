# Third-party notices

TessScope uses third-party software and model assets under their respective licenses.

## InstanSeg

The InstanSeg software and `single_channel_nuclei` model bundle are distributed under Apache-2.0. The release model card states that its training sources include CPDMI 2023 (CC BY 4.0), DSB 2018 / BBBC038 (CC0), and S-BSST265 (CC0). Exact source, asset, license, training-data links, and hashes are recorded in `data/manifests/instanseg-v0.1.2.json`.

TessScope's BBBC039 evaluation is decontaminated against BBBC038 before test scoring because BBBC038 is disclosed as observer training data.

## Chromatix

Chromatix is MIT licensed and pinned to the exact commit recorded in `uv.lock` and `decision-log.md`.

This notice is informational and does not replace the upstream licenses or model card.
