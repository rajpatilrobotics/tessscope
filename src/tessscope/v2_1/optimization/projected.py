"""Segmentation-primary projection for conflicting exact branch gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectedGradient:
    """Combined direction and transparent branch-conflict diagnostics."""

    combined: np.ndarray
    projected_focus: np.ndarray
    raw_dot: float
    projection_scale: float


def segmentation_primary_gradient(
    segmentation_gradient: np.ndarray,
    normalized_focus_gradient: np.ndarray,
    *,
    balance: float,
) -> ProjectedGradient:
    """Remove focus opposition to segmentation, then norm-balance the branches."""
    segmentation = np.asarray(segmentation_gradient, dtype=np.float64)
    focus = np.asarray(normalized_focus_gradient, dtype=np.float64)
    if segmentation.shape != focus.shape or segmentation.ndim != 1:
        raise ValueError("Branch gradients must be same-shape vectors")
    if not np.isfinite(segmentation).all() or not np.isfinite(focus).all():
        raise ValueError("Branch gradients must be finite")
    if balance < 0 or not np.isfinite(balance):
        raise ValueError("Projection balance must be finite and nonnegative")
    raw_dot = float(np.dot(segmentation, focus))
    projected_focus = focus.copy()
    segmentation_squared_norm = float(np.dot(segmentation, segmentation))
    if raw_dot < 0 and segmentation_squared_norm > 0:
        projected_focus -= raw_dot / segmentation_squared_norm * segmentation
    focus_norm = float(np.linalg.norm(projected_focus))
    if focus_norm <= 1e-12:
        projection_scale = 0.0
    else:
        projection_scale = (
            balance * float(np.linalg.norm(segmentation)) / focus_norm
        )
    return ProjectedGradient(
        combined=segmentation + projection_scale * projected_focus,
        projected_focus=projected_focus,
        raw_dot=raw_dot,
        projection_scale=projection_scale,
    )
