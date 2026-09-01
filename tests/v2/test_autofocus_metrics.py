"""Signed autofocus metric tests."""

from __future__ import annotations

import numpy as np
import pytest

from tessscope.v2.autofocus.metrics import focus_metrics


def test_focus_metrics_exclude_zero_only_from_direction() -> None:
    metrics = focus_metrics(
        np.asarray([-4.0, 0.0, 4.0]),
        np.asarray([-3.0, 1.0, 5.0]),
    )

    assert metrics["sample_count"] == 3
    assert metrics["nonzero_sample_count"] == 2
    assert metrics["signed_direction_accuracy"] == 1.0
    assert metrics["mae_um"] == 1.0
    assert metrics["correction_improvement_um"] > 0


def test_focus_metrics_respect_stage_action_bound() -> None:
    metrics = focus_metrics(np.asarray([6.0]), np.asarray([20.0]), stage_bound_um=6.0)

    assert metrics["mean_absolute_depth_after_correction_um"] == 0.0


def test_focus_metrics_require_nonzero_depth() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        focus_metrics(np.zeros(2), np.ones(2))
