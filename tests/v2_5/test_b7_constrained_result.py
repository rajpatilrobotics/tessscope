"""Frozen assertions for the completed v2.5 B7 continuity matrix."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_5"
    / "optimization"
    / "b7-constrained.json"
)


def test_v2_5_b7_matrix_exhausted_primary_and_alternate_budgets() -> None:
    result = json.loads(RESULT.read_text())
    assert result["basis"] == "B7"
    assert result["budget"]["starts"] == [
        "b7_v2_4_balanced_step14",
        "b7_v2_2_piecewise_028",
    ]
    assert result["budget"]["residual_bounds"] == [0.075, 0.065, 0.055]
    assert len(result["primary_runs"]) == 6
    assert sum(row["steps"] for row in result["primary_runs"]) == 108
    assert result["primary_soft_eligible_names"] == []
    assert result["budget"]["alternate_activated"] is True
    assert len(result["alternate_runs"]) == 3
    assert all(row["iterations"] == 8 for row in result["alternate_runs"])


def test_v2_5_b7_result_freezes_zero_promotions_and_test_seal() -> None:
    result = json.loads(RESULT.read_text())
    assert result["selection"] is None
    assert result["selection_rows"] == []
    assert result["selected_for_derivative_piecewise_and_hard"] == []
    assert result["test_accessed"] is False
    for row in result["primary_runs"] + result["alternate_runs"]:
        assert row["training_first_scaled_violation"] > 0.0
        assert row["training_residual_scaled_violation"] > 0.0
    for row in result["alternate_runs"]:
        assert row["success"] is False
        assert row["status"] == 9
