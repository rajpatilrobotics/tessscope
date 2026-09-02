"""Frozen assertions for the completed v2.6 oracle/controller audit."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "diagnostics"
    / "oracle-controller-audit.json"
)


def test_v2_6_oracle_audit_is_paired_and_test_sealed() -> None:
    result = json.loads(RESULT.read_text())
    assert result["status"] == "complete_historical_validation_diagnostic_only"
    assert result["test_accessed"] is False
    assert result["current_candidate"]["summary"]["sample_count"] == 162
    assert result["zero_residual_oracles"]["candidate"]["sample_count"] == 162
    assert result["controller_grid"]["evaluated_per_system"] == 125
    assert result["controller_grid"]["best_candidate_vs_best_baseline"][
        "well_count"
    ] == 27


def test_v2_6_oracle_rejects_controller_only_route() -> None:
    result = json.loads(RESULT.read_text())
    oracle = result["zero_residual_oracles"]["candidate"]
    assert oracle["corrected_pq"] < result["minimum_corrected_pq_from_actual_baseline"]
    assert result["controller_route_supported_before_exposure_audit"] is False
    assert not any(result["route_checks_before_exposure_audit"].values())
