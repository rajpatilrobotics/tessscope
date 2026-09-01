"""Pure helpers for the TessScope v2.1 near-miss audit."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr

from tessscope.v2.autofocus.metrics import focus_metrics

HARD_METRICS = (
    "panoptic_quality",
    "segmentation_quality",
    "recognition_quality",
    "foreground_dice",
    "absolute_count_error",
)


def mean_metrics(rows: Iterable[dict], metrics: tuple[str, ...] = HARD_METRICS) -> dict:
    """Return finite arithmetic means for the requested row metrics."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot summarize an empty metric row collection")
    return {
        metric: float(np.mean([float(row[metric]) for row in materialized])) for metric in metrics
    }


def summarize_hard_rows(rows: Iterable[dict]) -> dict:
    """Summarize one candidate's hard segmentation and signed-focus rows."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("A candidate must have at least one hard-validation row")
    off_focus = [row for row in materialized if float(row["depth_um"]) != 0.0]
    hard_off_focus = [row for row in off_focus if bool(row["hard_dense_patch"])]
    focus = focus_metrics(
        np.asarray([row["depth_um"] for row in materialized], dtype=np.float64),
        np.asarray([row["predicted_depth_um"] for row in materialized], dtype=np.float64),
    )
    return {
        "all_off_focus": mean_metrics(off_focus),
        "hard_dense_off_focus": mean_metrics(hard_off_focus),
        "focus": focus,
    }


def group_summary(rows: Iterable[dict], key: str) -> list[dict]:
    """Aggregate hard/focus metrics by a stable categorical key."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    result = []
    for value in sorted(groups):
        group_rows = groups[value]
        off_focus = [row for row in group_rows if float(row["depth_um"]) != 0.0]
        nonzero = off_focus or group_rows
        true_depth = np.asarray([row["depth_um"] for row in nonzero], dtype=np.float64)
        predicted = np.asarray([row["predicted_depth_um"] for row in nonzero], dtype=np.float64)
        direction = true_depth != 0
        result.append(
            {
                key: value,
                "row_count": len(group_rows),
                **mean_metrics(nonzero),
                "focus_mae_um": float(np.mean(np.abs(predicted - true_depth))),
                "signed_direction_accuracy": (
                    float(np.mean(np.sign(predicted[direction]) == np.sign(true_depth[direction])))
                    if np.any(direction)
                    else None
                ),
            }
        )
    return result


def paired_comparison(
    reference_rows: Iterable[dict],
    candidate_rows: Iterable[dict],
    *,
    group_key: str,
) -> list[dict]:
    """Compare two row sets after enforcing identical field/depth membership."""
    reference = {
        (str(row["field_id"]), float(row["depth_um"])): row
        for row in reference_rows
        if float(row["depth_um"]) != 0.0
    }
    candidate = {
        (str(row["field_id"]), float(row["depth_um"])): row
        for row in candidate_rows
        if float(row["depth_um"]) != 0.0
    }
    if reference.keys() != candidate.keys():
        raise ValueError("Paired candidate rows do not share field/depth membership")
    grouped: dict[str, list[tuple[dict, dict]]] = {}
    for row_key in sorted(reference):
        reference_row = reference[row_key]
        candidate_row = candidate[row_key]
        value = str(candidate_row[group_key])
        if str(reference_row[group_key]) != value:
            raise ValueError(f"Paired rows disagree on {group_key}")
        grouped.setdefault(value, []).append((reference_row, candidate_row))
    output = []
    for value in sorted(grouped):
        pairs = grouped[value]
        row = {group_key: value, "paired_row_count": len(pairs)}
        for metric in HARD_METRICS:
            delta = [
                float(candidate[metric]) - float(reference[metric])
                for reference, candidate in pairs
            ]
            row[f"delta_{metric}"] = float(np.mean(delta))
        reference_error = [
            abs(float(reference["predicted_depth_um"]) - float(reference["depth_um"]))
            for reference, _ in pairs
        ]
        candidate_error = [
            abs(float(candidate["predicted_depth_um"]) - float(candidate["depth_um"]))
            for _, candidate in pairs
        ]
        row["delta_focus_absolute_error_um"] = float(
            np.mean(candidate_error) - np.mean(reference_error)
        )
        output.append(row)
    return output


def correlation_table(
    candidates: Iterable[dict],
    predictor_keys: tuple[str, ...],
    outcome_keys: tuple[str, ...],
) -> list[dict]:
    """Return Pearson and Spearman correlations across promoted candidates."""
    materialized = list(candidates)
    result = []
    for predictor in predictor_keys:
        for outcome in outcome_keys:
            x = np.asarray([row[predictor] for row in materialized], dtype=np.float64)
            y = np.asarray([row[outcome] for row in materialized], dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(finite) < 3 or np.ptp(x[finite]) == 0 or np.ptp(y[finite]) == 0:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(pearsonr(x[finite], y[finite]).statistic)
                spearman = float(spearmanr(x[finite], y[finite]).statistic)
            result.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "candidate_count": int(np.count_nonzero(finite)),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                }
            )
    return result
