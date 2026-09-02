"""Frozen v2.5 soft eligibility and diversity rules."""

from __future__ import annotations

import math

from tessscope.v2_3.closed_loop import nondominated_closed_loop_names


def is_soft_eligible(
    row: dict,
    *,
    first_segmentation_maximum: float,
    feasibility_tolerance: float = 1e-4,
) -> bool:
    """Apply every preregistered B11 soft and physical gate."""
    finite_keys = (
        "first_segmentation_loss",
        "final_segmentation_loss",
        "residual_mae_um",
        "normalized_residual_depth_squared",
        "start_residual_mae_um",
        "residual_bound",
        "physical_coefficient_norm_radians",
        "minimum_support_energy_fraction",
    )
    return bool(
        all(math.isfinite(float(row[key])) for key in finite_keys)
        and row["basis"] == "B11"
        and row["finite_parameters"]
        and row["first_segmentation_loss"] <= first_segmentation_maximum
        and row["normalized_residual_depth_squared"]
        <= row["residual_bound"] + feasibility_tolerance
        and row["residual_mae_um"] < row["start_residual_mae_um"]
        and row["physical_coefficient_norm_radians"] < 2.5
        and row["minimum_support_energy_fraction"] >= 0.995
    )


def selection_key(row: dict) -> tuple:
    return (
        row["final_segmentation_loss"],
        row["residual_mae_um"],
        row["first_segmentation_loss"],
        row["name"],
    )


def select_candidates(rows: list[dict], maximum: int = 3) -> dict:
    """Select hash-unique nondominated B11 points with start-family diversity."""
    eligible = [row for row in rows if row["eligible"]]
    canonical_by_hash: dict[str, dict] = {}
    for row in sorted(eligible, key=lambda item: item["name"]):
        canonical_by_hash.setdefault(row["parameter_sha256"], row)
    canonical = list(canonical_by_hash.values())
    nondominated_names = nondominated_closed_loop_names(canonical)
    nondominated = [row for row in canonical if row["name"] in nondominated_names]
    winner_by_start: dict[str, dict] = {}
    for row in sorted(nondominated, key=selection_key):
        winner_by_start.setdefault(row["start_name"], row)
    selected = sorted(winner_by_start.values(), key=selection_key)[:maximum]
    return {
        "eligible_names": sorted(row["name"] for row in eligible),
        "canonical_eligible_names": sorted(row["name"] for row in canonical),
        "nondominated_names": sorted(nondominated_names),
        "start_winner_names": sorted(row["name"] for row in winner_by_start.values()),
        "selected_names": [row["name"] for row in selected],
    }


def alternate_start_key(row: dict) -> tuple:
    """Training-only deterministic ranking for the conditional SLSQP alternate."""
    violation = max(0.0, row["training_first_scaled_violation"]) + max(
        0.0, row["training_residual_scaled_violation"]
    )
    return violation, row["training_final_segmentation_loss"], row["name"]
