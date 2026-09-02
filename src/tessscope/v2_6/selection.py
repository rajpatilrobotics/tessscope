"""Frozen soft eligibility and deterministic selection for v2.6 two-mask work."""

from __future__ import annotations

import numpy as np


def two_mask_physical(physical: dict) -> bool:
    """Return whether both masks pass the frozen B7 physical gates."""
    return bool(
        physical["sensing"]["physical_coefficient_norm_radians"] < 2.5
        and physical["capture"]["physical_coefficient_norm_radians"] < 2.5
        and physical["sensing"]["minimum_support_energy_fraction"] >= 0.995
        and physical["capture"]["minimum_support_energy_fraction"] >= 0.995
    )


def soft_eligible(
    metrics: dict,
    physical: dict,
    *,
    first_limit: float,
    residual_limit: float,
    baseline_final_limit: float | None = None,
) -> bool:
    """Apply the frozen first, focus, residual, baseline, finite, and physical gates."""
    values = np.asarray(list(metrics.values()), dtype=np.float64)
    baseline_passes = (
        True
        if baseline_final_limit is None
        else metrics["final_segmentation_loss"] < baseline_final_limit
    )
    return bool(
        np.isfinite(values).all()
        and metrics["first_segmentation_loss"] <= first_limit
        and metrics["normalized_residual_depth_squared"] <= residual_limit + 1e-4
        and metrics["residual_mae_um"] <= 1.0
        and baseline_passes
        and two_mask_physical(physical)
    )


def select_soft_rows(rows: list[dict], maximum: int) -> list[str]:
    """Select eligible rows by the preregistered deterministic ordering."""
    if maximum <= 0:
        raise ValueError("Selection maximum must be positive")
    return [
        row["name"]
        for row in sorted(
            (row for row in rows if row["eligible"]),
            key=lambda row: (
                row["metrics"]["final_segmentation_loss"],
                row["metrics"]["residual_mae_um"],
                row["metrics"]["first_segmentation_loss"],
                row["name"],
            ),
        )[:maximum]
    ]
