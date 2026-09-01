"""Pre-registered B7 piecewise-family generation and soft screening."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

PROJECTED_RADIUS_RADIANS = 2.4975
DEDUPLICATION_DECIMALS = 10


def project_coefficients(
    coefficients: np.ndarray,
    radius: float = PROJECTED_RADIUS_RADIANS,
) -> tuple[np.ndarray, bool]:
    """Radially project only vectors outside the frozen open-ball radius."""
    value = np.asarray(coefficients, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError(f"Expected seven B7 coefficients, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Piecewise coefficients must be finite")
    norm = float(np.linalg.norm(value))
    if norm > radius:
        return value * (radius / norm), True
    return value.copy(), False


def _decimal_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def generate_piecewise_family(
    segmentation_coefficients: np.ndarray,
    focus_coefficients: np.ndarray,
) -> list[dict]:
    """Generate and deduplicate the complete pre-registered B7 mixture family."""
    segmentation = np.asarray(segmentation_coefficients, dtype=np.float64)
    focus = np.asarray(focus_coefficients, dtype=np.float64)
    if segmentation.shape != (7,) or focus.shape != (7,):
        raise ValueError("Both source pupils must contain seven B7 coefficients")
    unique: dict[tuple[float, ...], dict] = {}

    def add(
        coefficients: np.ndarray,
        *,
        family: str,
        values: dict[str, float],
        anchor: bool = False,
        original_naive_sum: bool = False,
    ) -> None:
        projected, was_projected = project_coefficients(coefficients)
        key = tuple(np.round(projected, DEDUPLICATION_DECIMALS))
        alias = {
            "family": family,
            "values": values,
            "anchor": anchor,
            "original_naive_sum": original_naive_sum,
        }
        if key in unique:
            unique[key]["aliases"].append(alias)
            unique[key]["is_family_anchor"] |= anchor
            unique[key]["includes_original_naive_sum"] |= original_naive_sum
            return
        unique[key] = {
            "phase_coefficients": projected.tolist(),
            "phase_rms_radians": float(np.linalg.norm(projected)),
            "was_projected": was_projected,
            "is_family_anchor": anchor,
            "includes_original_naive_sum": original_naive_sum,
            "aliases": [alias],
        }

    add(
        segmentation + focus,
        family="original_naive_sum",
        values={"segmentation_weight": 1.0, "focus_weight": 1.0},
        anchor=True,
        original_naive_sum=True,
    )
    for t_value in _decimal_grid(0.0, 1.0, 0.05):
        add(
            (1.0 - t_value) * segmentation + t_value * focus,
            family="convex_interpolation",
            values={"t": t_value},
            anchor=t_value in {0.0, 1.0},
        )
    for strength in _decimal_grid(0.0, 1.5, 0.05):
        add(
            segmentation + strength * focus,
            family="focus_injection",
            values={"lambda": strength},
            anchor=strength in {0.0, 1.0, 1.5},
        )
    weights = _decimal_grid(0.0, 1.5, 0.25)
    for segmentation_weight in weights:
        for focus_weight in weights:
            if segmentation_weight == focus_weight == 0.0:
                continue
            add(
                segmentation_weight * segmentation + focus_weight * focus,
                family="nonnegative_two_weight",
                values={
                    "segmentation_weight": segmentation_weight,
                    "focus_weight": focus_weight,
                },
                anchor=(
                    segmentation_weight in {0.0, 1.5}
                    and focus_weight in {0.0, 1.5}
                ),
            )
    points = list(unique.values())
    for index, point in enumerate(points):
        point["name"] = f"v2_2-piecewise-{index:03d}"
    return points


def nondominated_names(rows: Iterable[dict]) -> set[str]:
    """Return points not dominated when both soft losses are minimized."""
    values = list(rows)
    names = set()
    for candidate in values:
        dominated = any(
            other["name"] != candidate["name"]
            and other["segmentation_loss"] <= candidate["segmentation_loss"]
            and other["focus_mse"] <= candidate["focus_mse"]
            and (
                other["segmentation_loss"] < candidate["segmentation_loss"]
                or other["focus_mse"] < candidate["focus_mse"]
            )
            for other in values
        )
        if not dominated:
            names.add(candidate["name"])
    return names


def select_hard_candidates(
    rows: list[dict],
    *,
    joint_segmentation_loss: float,
    joint_focus_mse: float,
    segmentation_guard: float = 0.004,
    focus_guard: float = 0.25,
) -> tuple[list[str], dict[str, list[str]]]:
    """Apply the frozen soft envelope, guard-band, anchor, and bracket rule."""
    if not rows:
        raise ValueError("Soft screening requires at least one piecewise point")
    reasons: dict[str, set[str]] = {row["name"]: set() for row in rows}
    nondominated = nondominated_names(rows)
    by_name = {row["name"]: row for row in rows}
    for name in nondominated:
        reasons[name].add("soft_nondominated")
    for row in rows:
        if row["is_family_anchor"]:
            reasons[row["name"]].add("family_anchor")
        if row["includes_original_naive_sum"]:
            reasons[row["name"]].add("original_naive_sum")
        if any(
            row["segmentation_loss"]
            <= by_name[frontier_name]["segmentation_loss"] + segmentation_guard
            and row["focus_mse"]
            <= by_name[frontier_name]["focus_mse"] + focus_guard
            for frontier_name in nondominated
        ):
            reasons[row["name"]].add("soft_envelope_guard_band")

    for metric, target in (
        ("segmentation_loss", joint_segmentation_loss),
        ("focus_mse", joint_focus_mse),
    ):
        ordered = sorted(rows, key=lambda row: (row[metric], row["name"]))
        lower = [row for row in ordered if row[metric] <= target][-2:]
        upper = [row for row in ordered if row[metric] >= target][:2]
        for row in [*lower, *upper]:
            reasons[row["name"]].add(f"joint_{metric}_bracket")
    selected = sorted(name for name, why in reasons.items() if why)
    frozen_reasons = {
        name: sorted(reasons[name])
        for name in selected
    }
    return selected, frozen_reasons
