"""Pure oracle, controller, and paired-evidence utilities for v2.6."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from tessscope.evaluation.metrics import grouped_bootstrap_difference


@dataclass(frozen=True)
class ControllerCalibration:
    """Small interpretable calibration applied to a signed depth prediction."""

    gain: float = 1.0
    bias_um: float = 0.0
    cubic: float = 0.0

    def calibrated_prediction(self, prediction_um: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction_um, dtype=np.float64)
        return (
            self.gain * prediction
            + self.bias_um
            + self.cubic * np.power(prediction, 3) / 36.0
        )

    def residual_depth(
        self, true_depth_um: np.ndarray, prediction_um: np.ndarray
    ) -> np.ndarray:
        calibrated = self.calibrated_prediction(prediction_um)
        action = np.clip(-calibrated, -6.0, 6.0)
        return np.asarray(true_depth_um, dtype=np.float64) + action

    def is_monotone(self) -> bool:
        probe = np.linspace(-6.0, 6.0, 241)
        derivative = self.gain + self.cubic * np.square(probe) / 12.0
        return bool(np.min(derivative) > 0.0)


def _row_key(row: dict, depth_key: str) -> tuple[str, str, float]:
    return row["well"], row["field_id"], float(row[depth_key])


def paired_well_evidence(
    candidate_rows: list[dict],
    reference_rows: list[dict],
    *,
    value_key: str = "after_pq",
    depth_key: str = "original_depth_um",
    seed: int = 20260902,
) -> dict:
    """Return paired grouped uncertainty and well-level stability."""
    candidate = {_row_key(row, depth_key): row for row in candidate_rows}
    reference = {_row_key(row, depth_key): row for row in reference_rows}
    if candidate.keys() != reference.keys() or not candidate:
        raise ValueError("Candidate/reference rows must be nonempty and exactly paired")
    keys = sorted(candidate)
    groups = np.asarray([key[0] for key in keys])
    candidate_values = np.asarray([candidate[key][value_key] for key in keys])
    reference_values = np.asarray([reference[key][value_key] for key in keys])
    bootstrap = grouped_bootstrap_difference(
        groups,
        candidate_values,
        reference_values,
        replicates=2000,
        seed=seed,
    )
    by_well: dict[str, list[float]] = defaultdict(list)
    for key, difference in zip(
        keys, candidate_values - reference_values, strict=True
    ):
        by_well[key[0]].append(float(difference))
    well_means = {
        well: float(np.mean(values)) for well, values in sorted(by_well.items())
    }
    wells = sorted(well_means)
    leave_one_out = {
        omitted: float(
            np.mean([well_means[well] for well in wells if well != omitted])
        )
        for omitted in wells
    }
    return {
        **bootstrap,
        "well_count": len(wells),
        "positive_well_fraction": float(
            np.mean([well_means[well] > 0.0 for well in wells])
        ),
        "well_mean_differences": well_means,
        "leave_one_well_out_mean_differences": leave_one_out,
        "minimum_leave_one_well_out_mean_difference": min(leave_one_out.values()),
    }


def _curve_index(
    rows: list[dict], design: str
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row["design"] == design:
            grouped[(row["well"], row["field_id"])].append(
                (float(row["depth_um"]), float(row["panoptic_quality"]))
            )
    result = {}
    for key, values in grouped.items():
        ordered = sorted(values)
        depths = np.asarray([value[0] for value in ordered])
        pq = np.asarray([value[1] for value in ordered])
        if not np.array_equal(depths, np.asarray([-6, -4, -2, 0, 2, 4, 6])):
            raise ValueError(f"Incomplete depth curve for {key}")
        result[key] = depths, pq
    return result


def interpolated_controller_rows(
    rows: list[dict],
    design: str,
    calibration: ControllerCalibration,
    *,
    hard_only: bool = True,
) -> list[dict]:
    """Estimate corrected PQ from each field's frozen seven-depth PQ curve."""
    curves = _curve_index(rows, design)
    selected = [
        row
        for row in rows
        if row["design"] == design
        and row["depth_um"] != 0.0
        and (not hard_only or row["hard_dense_patch"])
    ]
    result = []
    for row in selected:
        residual = float(
            calibration.residual_depth(
                np.asarray([row["depth_um"]]),
                np.asarray([row["predicted_depth_um"]]),
            )[0]
        )
        depths, pq = curves[(row["well"], row["field_id"])]
        result.append(
            {
                "design": design,
                "well": row["well"],
                "field_id": row["field_id"],
                "original_depth_um": float(row["depth_um"]),
                "residual_depth_um": residual,
                "before_pq": float(row["panoptic_quality"]),
                "after_pq": float(np.interp(residual, depths, pq)),
            }
        )
    return result


def oracle_rows(rows: list[dict], design: str, *, hard_only: bool = True) -> list[dict]:
    """Map every nonzero frame to its same-field frozen depth-zero PQ."""
    curves = _curve_index(rows, design)
    selected = [
        row
        for row in rows
        if row["design"] == design
        and row["depth_um"] != 0.0
        and (not hard_only or row["hard_dense_patch"])
    ]
    result = []
    for row in selected:
        depths, pq = curves[(row["well"], row["field_id"])]
        result.append(
            {
                "design": f"{design}-oracle-zero-residual",
                "well": row["well"],
                "field_id": row["field_id"],
                "original_depth_um": float(row["depth_um"]),
                "residual_depth_um": 0.0,
                "before_pq": float(row["panoptic_quality"]),
                "after_pq": float(pq[np.flatnonzero(depths == 0.0)[0]]),
            }
        )
    return result


def summarize_controller_rows(rows: list[dict]) -> dict:
    """Summarize corrected PQ, focus, direction, and frame reliability."""
    original = np.asarray([row["original_depth_um"] for row in rows])
    residual = np.asarray([row["residual_depth_um"] for row in rows])
    estimated = original - residual
    return {
        "sample_count": len(rows),
        "corrected_pq": float(np.mean([row["after_pq"] for row in rows])),
        "focus_mae_um": float(np.mean(np.abs(residual))),
        "signed_direction_accuracy": float(np.mean(np.sign(estimated) == np.sign(original))),
        "fraction_frames_improved": float(
            np.mean([row["after_pq"] > row["before_pq"] for row in rows])
        ),
    }


def controller_error_decomposition(rows: list[dict], design: str) -> dict:
    """Decompose the frozen controller's errors without changing it."""
    selected = [
        row for row in rows if row["design"] == design and row["depth_um"] != 0.0
    ]
    true = np.asarray([row["depth_um"] for row in selected], dtype=np.float64)
    predicted = np.asarray(
        [row["predicted_depth_um"] for row in selected], dtype=np.float64
    )
    residual = true - np.clip(predicted, -6.0, 6.0)

    def grouped_mean(group_values: list[str | float]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(group_values):
            grouped[str(value)].append(index)
        return {
            name: {
                "count": len(indices),
                "mean_error_um": float(np.mean(predicted[indices] - true[indices])),
                "mae_um": float(np.mean(np.abs(residual[indices]))),
            }
            for name, indices in sorted(grouped.items())
        }

    magnitude = [
        "near" if abs(value) <= 2 else "mid" if abs(value) <= 4 else "far"
        for value in true
    ]
    density = ["hard" if row["hard_dense_patch"] else "other" for row in selected]
    linear_gain, linear_bias = np.polyfit(true, predicted, deg=1)
    cubic_fit = np.polyfit(true, predicted, deg=3)
    return {
        "count": len(selected),
        "mean_prediction_bias_um": float(np.mean(predicted - true)),
        "mae_um": float(np.mean(np.abs(residual))),
        "signed_direction_accuracy": float(np.mean(np.sign(predicted) == np.sign(true))),
        "action_saturation_fraction": float(np.mean(np.abs(predicted) >= 6.0)),
        "linear_fit_prediction_from_true": {
            "gain": float(linear_gain),
            "bias_um": float(linear_bias),
        },
        "cubic_fit_prediction_from_true": [float(value) for value in cubic_fit],
        "by_true_depth_um": grouped_mean([float(value) for value in true]),
        "by_sign": grouped_mean(["positive" if value > 0 else "negative" for value in true]),
        "by_magnitude": grouped_mean(magnitude),
        "by_density": grouped_mean(density),
        "by_well": grouped_mean([row["well"] for row in selected]),
    }


def controller_grid() -> list[ControllerCalibration]:
    """Return the exact preregistered monotone calibration grid."""
    result = []
    for gain in (0.85, 0.925, 1.0, 1.075, 1.15):
        for bias in (-0.5, -0.25, 0.0, 0.25, 0.5):
            for cubic in (-0.2, -0.1, 0.0, 0.1, 0.2):
                calibration = ControllerCalibration(gain, bias, cubic)
                if calibration.is_monotone():
                    result.append(calibration)
    return result
