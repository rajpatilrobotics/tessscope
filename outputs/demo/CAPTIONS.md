# TessScope demo captions and alt text

All evidence is validation-only. PQ is panoptic quality, not percent accuracy. The locked test remains sealed, and no physical microscope has validated the system.

## Hero Evidence

**Caption:** Frozen representative validation field n21_s1 at −2 µm for the exact-gradient design, showing the first sensor frame, corrected sensor frame, B7 pupil phase, and Chromatix sensor-plane PSF. The microscopy panels use the documented global display transform; this individual frame is not the paired population result.

**Alt text:** Four panels from frozen validation evidence show the exact-gradient first frame with automated-reference and InstanSeg boundaries, the corrected frame after one stage action, the exact B7 pupil phase, and its minus-two-micrometre PSF.

## Causal Comparison

**Caption:** The exact design increases its own hard off-focus PQ from 0.4922 to 0.5523 after one predicted stage action. The causal comparison against its forward-identical stopped-gradient control is +0.0063 PQ with a 95% well-bootstrap interval of +0.0007 to +0.0115. The comparison against piecewise-028 is only +0.0013 with an interval from −0.0017 to +0.0048, so superiority over piecewise is not supported.

**Alt text:** A paired chart shows exact-design PQ rising from 0.4922 before correction to 0.5523 after correction. A forest plot shows a positive confidence interval for exact versus stopped gradients and a confidence interval crossing zero for exact versus piecewise.

## Matched Microscopy

**Caption:** Representative hard-density validation field n21_s1 at −2 µm, selected before rendering by the frozen median-distance rule. Every system uses the identical crop, geometry, nearest-neighbor label interpolation, and global observer transform. Blue boundaries are automated BBBC006/CellProfiler references; orange boundaries are official InstanSeg outputs. The representative frame is not the population effect.

**Alt text:** Two-row matched microscopy grid for validation field n21_s1. The top row compares clear, exact-gradient, stopped-gradient, and piecewise first frames at minus two micrometres. The bottom row shows the automated reference and corrected exact, stopped, and piecewise frames with reference and InstanSeg boundaries.

## Pq Focus Depth

**Caption:** Depth-resolved validation behavior over 27 hard-density wells. Corrected PQ remains near 0.55 across exact, stopped, and piecewise systems while exact autofocus retains lower residual defocus than its stopped-gradient control. Bands are grouped well bootstrap intervals and do not treat frames as independent replicates.

**Alt text:** Two line charts across six nonzero defocus depths. The left chart shows first-frame and corrected panoptic quality. The right chart shows absolute residual defocus for exact, stopped, and piecewise systems, with 95 percent well-bootstrap bands.

## Pupil Psf Depth

**Caption:** Wrapped phase masks and unit-energy Chromatix/JAX sensor-plane PSFs for clear, exact-gradient, forward-identical stopped-gradient, and piecewise-028 systems. PSFs span the complete frozen −6 to +6 µm depth grid with one global log-intensity scale and identical central crop.

**Alt text:** Four-row optical comparison. Each row begins with a circular wrapped phase mask and continues with seven PSF images from minus six through plus six micrometres. Clear, exact, stopped, and piecewise masks create visibly different depth-dependent PSFs.

## Gradient Architecture

**Caption:** Three explicit Tesseract component APIs connect JAX/Chromatix optics, NumPy/SciPy autofocus, and PyTorch/InstanSeg. The optics API is called for the first depth stack and again at autofocus-predicted residual depths; the observer scores first and final frames. Tesseract carries the exact VJP across these runtime boundaries. The stopped control keeps every forward call identical but removes the stage-action VJP edge.

**Alt text:** A left-to-right served-call pipeline shows two JAX Chromatix optics calls, one NumPy SciPy autofocus call, and one PyTorch InstanSeg observer call, each inside a labeled Tesseract API boundary. A blue exact VJP crosses every boundary; the dashed stopped control ends at the stage-action learning edge while the forward path stays identical.

## Validation depth sweep

**Caption:** Cached replay of the frozen representative validation field across the complete seven-depth grid. Exact, stopped-gradient, and piecewise systems use identical display and label-overlay rules before and after one stage action.

**Alt text:** An animation steps from minus six to plus six micrometres of initial defocus. Three rows compare exact, stopped-gradient, and piecewise first and corrected frames with automated-reference and InstanSeg boundaries.

MP4: `outputs/demo/replay/validation-depth-sweep.mp4`

GIF: `outputs/demo/replay/validation-depth-sweep.gif`
