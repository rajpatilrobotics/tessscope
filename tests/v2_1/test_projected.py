"""Tests for segmentation-primary exact-gradient projection."""

import numpy as np
import pytest

from tessscope.v2_1.optimization.projected import segmentation_primary_gradient


def test_conflicting_focus_component_becomes_orthogonal() -> None:
    segmentation = np.asarray([1.0, 0.0])
    focus = np.asarray([-1.0, 2.0])
    result = segmentation_primary_gradient(
        segmentation,
        focus,
        balance=0.5,
    )
    assert result.raw_dot == pytest.approx(-1.0)
    assert np.dot(segmentation, result.projected_focus) == pytest.approx(0.0)
    assert np.dot(segmentation, result.combined) > 0


def test_aligned_focus_gradient_is_not_projected() -> None:
    segmentation = np.asarray([1.0, 0.0])
    focus = np.asarray([1.0, 2.0])
    result = segmentation_primary_gradient(
        segmentation,
        focus,
        balance=1.0,
    )
    np.testing.assert_allclose(result.projected_focus, focus)
    assert result.raw_dot == pytest.approx(1.0)


def test_projection_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same-shape"):
        segmentation_primary_gradient(
            np.asarray([1.0]),
            np.asarray([1.0, 2.0]),
            balance=1.0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        segmentation_primary_gradient(
            np.asarray([1.0]),
            np.asarray([1.0]),
            balance=-1.0,
        )
