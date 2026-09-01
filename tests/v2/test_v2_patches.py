"""Real training-field preparation checks without touching locked test wells."""

from __future__ import annotations

import numpy as np

from tessscope.v2.data.bbbc006 import records_for_split
from tessscope.v2.data.patches import patch_origins, prepare_patch
from tessscope.v2.data.preprocess import load_global_normalization


def test_prepare_real_training_patch_has_frozen_shapes() -> None:
    record = records_for_split("training")[0]
    patch = prepare_patch(
        record,
        patch_origins("training")[0],
        load_global_normalization(),
    )

    assert patch.split == "training"
    assert patch.object_image.shape == (256, 256)
    assert patch.object_image.dtype == np.float32
    assert patch.instance_labels.shape == (330, 330)
    assert patch.centers_yx.shape == (32, 2)
    assert patch.valid_objects.shape == (32,)
    assert int(np.count_nonzero(patch.valid_objects)) <= 32


def test_patch_origins_explicitly_reject_test() -> None:
    try:
        patch_origins("test")
    except ValueError as error:
        assert "exclude" in str(error)
    else:
        raise AssertionError("Locked test patch origins must not be available during tuning")
