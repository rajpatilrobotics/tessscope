"""Signed-depth and stage-action metrics with explicit zero-depth handling."""

from __future__ import annotations

import numpy as np


def focus_metrics(
    true_depth_um: np.ndarray,
    predicted_depth_um: np.ndarray,
    *,
    stage_bound_um: float = 6.0,
) -> dict[str, float | int]:
    """Measure signed focus and the residual after one clipped stage correction."""
    true_depth = np.asarray(true_depth_um, dtype=np.float64)
    predicted = np.asarray(predicted_depth_um, dtype=np.float64)
    if true_depth.shape != predicted.shape or true_depth.ndim != 1:
        raise ValueError("Focus metric inputs must be equal one-dimensional arrays")
    if len(true_depth) == 0 or not np.isfinite(true_depth).all() or not np.isfinite(
        predicted
    ).all():
        raise ValueError("Focus metric inputs must be non-empty and finite")
    nonzero = true_depth != 0
    if not np.any(nonzero):
        raise ValueError("Signed direction requires at least one nonzero target depth")
    stage_action = np.clip(-predicted, -stage_bound_um, stage_bound_um)
    residual = true_depth + stage_action
    return {
        "sample_count": len(true_depth),
        "nonzero_sample_count": int(np.count_nonzero(nonzero)),
        "signed_direction_accuracy": float(
            np.mean(np.sign(predicted[nonzero]) == np.sign(true_depth[nonzero]))
        ),
        "mae_um": float(np.mean(np.abs(predicted - true_depth))),
        "nonzero_mae_um": float(
            np.mean(np.abs(predicted[nonzero] - true_depth[nonzero]))
        ),
        "median_absolute_error_um": float(np.median(np.abs(predicted - true_depth))),
        "mean_absolute_depth_before_correction_um": float(np.mean(np.abs(true_depth))),
        "mean_absolute_depth_after_correction_um": float(np.mean(np.abs(residual))),
        "correction_improvement_um": float(
            np.mean(np.abs(true_depth)) - np.mean(np.abs(residual))
        ),
    }
