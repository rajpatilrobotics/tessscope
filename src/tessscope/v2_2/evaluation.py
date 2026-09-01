"""Matched-frontier statistics for TessScope v2.2."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def hard_pareto_names(summaries: dict[str, dict]) -> set[str]:
    """Return designs not dominated when PQ is maximized and focus MAE minimized."""
    names = set()
    for name, candidate in summaries.items():
        dominated = any(
            other_name != name
            and other["hard_dense_off_focus_pq"]
            >= candidate["hard_dense_off_focus_pq"]
            and other["mae_um"] <= candidate["mae_um"]
            and (
                other["hard_dense_off_focus_pq"]
                > candidate["hard_dense_off_focus_pq"]
                or other["mae_um"] < candidate["mae_um"]
            )
            for other_name, other in summaries.items()
        )
        if not dominated:
            names.add(name)
    return names


def hypervolume_2d(points: Iterable[tuple[float, float]]) -> float:
    """Area dominated by maximize/maximize points relative to the frozen origin."""
    positive = sorted(
        {(max(float(x), 0.0), max(float(y), 0.0)) for x, y in points}
    )
    if not positive:
        return 0.0
    area = 0.0
    previous_x = 0.0
    for index, (x_value, _) in enumerate(positive):
        if x_value <= previous_x:
            continue
        best_y = max(y for _, y in positive[index:])
        area += (x_value - previous_x) * best_y
        previous_x = x_value
    return float(area)


def paired_focus_mae_bootstrap(
    rows: list[dict],
    candidate: str,
    reference: str,
    *,
    replicates: int = 2000,
    seed: int = 20260901,
) -> dict:
    """Bootstrap frozen all-depth reference-minus-candidate MAE by well."""

    def per_well(design: str) -> dict[str, float]:
        design_rows = [row for row in rows if row["design"] == design]
        wells = sorted({row["well"] for row in design_rows})
        return {
            well: float(
                np.mean(
                    [
                        abs(row["predicted_depth_um"] - row["depth_um"])
                        for row in design_rows
                        if row["well"] == well
                    ]
                )
            )
            for well in wells
        }

    candidate_values = per_well(candidate)
    reference_values = per_well(reference)
    if candidate_values.keys() != reference_values.keys() or not candidate_values:
        raise ValueError("Focus rows are not paired by well")
    wells = sorted(candidate_values)
    differences = np.asarray(
        [reference_values[well] - candidate_values[well] for well in wells]
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(wells), size=(replicates, len(wells)))
    bootstrap = np.mean(differences[samples], axis=1)
    return {
        "source_count": len(wells),
        "replicates": replicates,
        "mean_difference": float(np.mean(differences)),
        "ci_lower_95": float(np.quantile(bootstrap, 0.025)),
        "ci_upper_95": float(np.quantile(bootstrap, 0.975)),
        "candidate": candidate,
        "reference": reference,
        "definition": "reference focus MAE minus candidate focus MAE",
    }
