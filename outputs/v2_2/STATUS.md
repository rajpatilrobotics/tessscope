# TessScope v2.2 matched-frontier status

## Outcome

V2.2 is complete as a negative validation-only experiment. The frozen v2.1 B7 joint
pupil does not lie beyond a strong, pre-registered piecewise frontier built from the
matched B7 segmentation-only and focus-only pupils. The BBBC006 locked test remained
sealed.

## Fair baseline

The family contains convex interpolation, dense focus injection, nonnegative two-weight
mixtures, and the original naive sum. Its 101 aliases collapse to 87 unique physical
pupils; 38 points survived the predeclared soft screen and all 38 received the same
45-well hard endpoint and one predicted stage correction.

The new frontier materially improves the old baseline. The original naive sum has hard
PQ `0.454014` and MAE `0.904854 µm`; another two-weight point reaches `0.472582` and
`0.868002 µm`, improving both.

## Matched result

| Design | Hard PQ | Focus MAE | Corrected hard PQ |
|---|---:|---:|---:|
| Frozen B7 joint | 0.496064 | 1.210735 µm | 0.552058 |
| Piecewise-028 | 0.502543 | 1.207179 µm | 0.551010 |

Piecewise-028 slightly dominates the joint on both first-frame coordinates. The paired
matched-focus PQ difference (joint minus piecewise) is `-0.00648`, with 95% interval
`[-0.01277, -0.00068]`. At matched segmentation, the joint focus effect is also below
the predeclared `0.10 µm` requirement. Adding the joint to the normalized piecewise
frontier contributes zero hypervolume.

## Decision

No v2.2 pupil is promoted, no matched B7 derivative-free search is warranted, and the
locked test is not accessed. The negative freeze is in
`configs/v2_2/pretest-block.json`. Per the approved fallback, work continues in the
separately named v2.3 closed-loop differentiable microscope experiment.
