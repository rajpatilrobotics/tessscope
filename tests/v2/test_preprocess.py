"""Global preprocessing contract checks."""

from __future__ import annotations

import numpy as np

from tessscope.v2.data.preprocess import GlobalNormalization, center_crop


def test_global_normalization_uses_one_affine_transform() -> None:
    images = np.asarray([[[10.0, 14.0]], [[18.0, 22.0]]])
    transform = GlobalNormalization(offset=10.0, scale=4.0)

    normalized = transform.apply(images)

    assert np.array_equal(normalized, np.asarray([[[0.0, 1.0]], [[2.0, 3.0]]]))


def test_center_crop_is_deterministic_for_stacks() -> None:
    stack = np.arange(2 * 8 * 10).reshape(2, 8, 10)

    crop = center_crop(stack, size=4)

    assert crop.shape == (2, 4, 4)
    assert np.array_equal(crop, stack[:, 2:6, 3:7])
