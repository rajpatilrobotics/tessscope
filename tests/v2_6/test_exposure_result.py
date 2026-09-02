"""Frozen assertions for the completed v2.6 exposure audit."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "configs" / "v2_6" / "exposure-audit.yaml"
RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "diagnostics"
    / "exposure-audit.json"
)


def test_v2_6_exposure_result_matches_contract_and_training_partitions() -> None:
    result = json.loads(RESULT.read_text())
    assert result["contract_sha256"] == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert len(result["rows"]) == 1440
    assert result["test_accessed"] is False
    assert set(result["patch_registry"]) == {"development", "confirmation"}
    frozen = json.loads(
        (
            PROJECT_ROOT
            / "data"
            / "manifests"
            / "v2_6"
            / "training-development-confirmation.json"
        ).read_text()
    )
    for partition, patches in result["patch_registry"].items():
        assert len(patches) == 4
        assert {row["well"] for row in patches} <= set(frozen["wells"][partition])


def test_v2_6_exposure_result_rejects_controller_exposure_route() -> None:
    result = json.loads(RESULT.read_text())
    systems = result["systems"]
    candidate = systems["v2_3-exact-balanced-segmentation_only-step-14"]
    baseline = systems["v2_2-piecewise-028"]
    assert candidate["development_selection"]["selected_first_fraction"] == 0.65
    assert baseline["development_selection"]["selected_first_fraction"] == 0.65
    assert result["candidate_vs_baseline_confirmation"]["mean_difference"] < 0.0
    assert result["controller_exposure_route_supported"] is False
    assert result["route_checks"]["candidate_first_pq_preserved"] is True
    assert sum(result["route_checks"].values()) == 1
