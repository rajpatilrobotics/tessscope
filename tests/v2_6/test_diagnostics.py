"""Pure tests for v2.6 oracle and controller diagnostics."""

import numpy as np

from tessscope.v2_6.diagnostics import (
    ControllerCalibration,
    controller_grid,
    interpolated_controller_rows,
    oracle_rows,
    paired_well_evidence,
    summarize_controller_rows,
)


def toy_rows(design: str) -> list[dict]:
    rows = []
    for well, offset in (("a01", 0.0), ("b01", 0.1)):
        for depth in (-6, -4, -2, 0, 2, 4, 6):
            rows.append(
                {
                    "design": design,
                    "well": well,
                    "field_id": f"{well}_s1",
                    "depth_um": float(depth),
                    "predicted_depth_um": float(depth),
                    "hard_dense_patch": True,
                    "panoptic_quality": 1.0 - abs(depth) / 10.0 + offset,
                }
            )
    return rows


def test_controller_calibration_preserves_identity_and_clips_action() -> None:
    identity = ControllerCalibration()
    true = np.asarray([-6.0, -2.0, 2.0, 6.0])
    assert np.allclose(identity.residual_depth(true, true), 0.0)
    biased = ControllerCalibration(gain=1.15, bias_um=0.5, cubic=0.2)
    residual = biased.residual_depth(np.asarray([6.0]), np.asarray([20.0]))
    assert np.allclose(residual, 0.0)
    assert biased.is_monotone()
    assert all(row.is_monotone() for row in controller_grid())


def test_oracle_and_identity_interpolation_reach_focus_curve() -> None:
    rows = toy_rows("candidate")
    oracle = oracle_rows(rows, "candidate")
    identity = interpolated_controller_rows(rows, "candidate", ControllerCalibration())
    assert oracle == [
        {**row, "design": "candidate-oracle-zero-residual"} for row in identity
    ]
    summary = summarize_controller_rows(identity)
    assert np.isclose(summary["corrected_pq"], 1.05)
    assert summary["focus_mae_um"] == 0.0
    assert summary["signed_direction_accuracy"] == 1.0
    assert summary["fraction_frames_improved"] == 1.0


def test_paired_evidence_groups_whole_wells_and_requires_exact_pairing() -> None:
    candidate = oracle_rows(toy_rows("candidate"), "candidate")
    reference = [{**row, "after_pq": row["after_pq"] - 0.01} for row in candidate]
    evidence = paired_well_evidence(candidate, reference)
    assert np.isclose(evidence["mean_difference"], 0.01)
    assert evidence["ci_lower_95"] > 0.0
    assert evidence["positive_well_fraction"] == 1.0
    assert evidence["minimum_leave_one_well_out_mean_difference"] > 0.0
