# TessScope v2 source notes

- Official BBBC006 page: https://bbbc.broadinstitute.org/BBBC006
- BBBC image-set index: https://bbbc.broadinstitute.org/image_sets
- Official segmentation pipeline: https://data.broadinstitute.org/bbbc/BBBC006/Batch_data.cppipe
- Recommended collection citation: Ljosa et al., *Nature Methods* (2012).
- Focus-quality precedent: Yang et al., “Assessing microscope image focus quality with
  deep learning,” *BMC Bioinformatics* (2018), using BBBC006.

The official page describes one 384-well U2OS plate, two sites per well, Hoechst and
phalloidin channels, a 20×/NA 0.45 system, 2× binning, 696×520 16-bit TIFFs, laser focus
at z16, and 2 µm axial spacing. The v2 sign convention is a documented coordinate
choice: `depth_um = 2 * (z - 16)` and `stage_action_um = -predicted_depth_um`.

Reference PNGs are automated CellProfiler instance labels produced from the z16 image.
They are not manual ground truth. The official pipeline uses global two-class Otsu,
weighted-variance minimization, intensity-based declumping, 20–100 px object diameter,
and does not discard border objects.
