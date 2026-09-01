"""Joint batch hygiene and shape checks."""

from __future__ import annotations

import numpy as np

from tessscope.v2.optimization.served import (
    collect_patches,
    materialize_joint_batch,
)


def test_materialized_joint_batch_has_two_supports_and_one_query() -> None:
    patches = collect_patches("training", 3, seed=47)
    batch = materialize_joint_batch(patches, noise_seed=47)

    assert batch.objects.shape == (3, 256, 256)
    assert batch.instance_labels.shape == (1, 330, 330)
    assert batch.centers_yx.shape == (1, 32, 2)
    assert batch.valid_objects.shape == (1, 32)
    assert batch.noise_standard_normal.shape == (3, 7, 256, 256)
    assert np.isfinite(batch.noise_standard_normal).all()
