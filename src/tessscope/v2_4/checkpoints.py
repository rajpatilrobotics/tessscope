"""Pure eligibility, diversity, and trajectory rules frozen for v2.4."""

from __future__ import annotations

import math

from tessscope.v2_3.closed_loop import nondominated_closed_loop_names

FIRST_SEGMENTATION_MAXIMUM = 1.098270310640335
SELECTION_METRICS = (
    "first_segmentation_loss",
    "final_segmentation_loss",
    "residual_mae_um",
)


def is_soft_eligible(row: dict) -> bool:
    """Apply the preregistered soft and physical eligibility rules."""
    finite_values = all(
        math.isfinite(float(row[key]))
        for key in (
            "objective",
            "first_segmentation_loss",
            "final_segmentation_loss",
            "focus_mse",
            "residual_mae_um",
            "normalized_residual_depth_squared",
            "normalized_stage_action_squared",
            "first_photon_mean",
            "second_photon_mean",
            "physical_coefficient_norm_radians",
            "minimum_support_energy_fraction",
        )
    )
    return bool(
        finite_values
        and row["seven_finite_phase_parameters"]
        and row["physical_coefficient_norm_radians"] < 2.5
        and row["minimum_support_energy_fraction"] >= 0.995
        and row["first_segmentation_loss"] <= FIRST_SEGMENTATION_MAXIMUM
        and row["residual_mae_um"] < row["start_residual_mae_um"]
    )


def selection_key(row: dict) -> tuple:
    """Return the exact frozen checkpoint tie-break order."""
    return (
        row["final_segmentation_loss"],
        row["residual_mae_um"],
        row["first_segmentation_loss"],
        row["name"],
    )


def select_checkpoints(rows: list[dict], maximum: int = 3) -> dict:
    """Select globally nondominated, hash-unique, one-per-run checkpoints."""
    eligible = [row for row in rows if row["eligible"]]
    canonical_by_hash = {}
    for row in sorted(eligible, key=lambda item: item["name"]):
        canonical_by_hash.setdefault(row["parameter_sha256"], row)
    canonical = list(canonical_by_hash.values())
    nondominated_names = nondominated_closed_loop_names(canonical)
    nondominated = [row for row in canonical if row["name"] in nondominated_names]
    winner_by_run = {}
    for row in sorted(nondominated, key=selection_key):
        winner_by_run.setdefault(row["source_run"], row)
    selected = sorted(winner_by_run.values(), key=selection_key)[:maximum]
    return {
        "eligible_names": sorted(row["name"] for row in eligible),
        "canonical_eligible_names": sorted(row["name"] for row in canonical),
        "nondominated_names": sorted(nondominated_names),
        "run_winner_names": sorted(row["name"] for row in winner_by_run.values()),
        "selected_names": [row["name"] for row in selected],
    }


def eligible_intervals(rows: list[dict]) -> list[list[int]]:
    """Group a run's eligible checkpoint steps into inclusive intervals."""
    steps = sorted(int(row["step"]) for row in rows if row["eligible"])
    if not steps:
        return []
    intervals = []
    first = previous = steps[0]
    for step in steps[1:]:
        if step != previous + 1:
            intervals.append([first, previous])
            first = step
        previous = step
    intervals.append([first, previous])
    return intervals
