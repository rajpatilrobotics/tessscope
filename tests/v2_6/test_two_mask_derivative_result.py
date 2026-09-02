"""Frozen-result checks for the v2.6 sequential-mask derivative gate."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "gates"
    / "two-mask-derivative.json"
)


def test_two_mask_derivative_passes_every_registered_direction() -> None:
    result = json.loads(RESULT.read_text())
    assert result["passed_v2_6"] is True
    assert set(result["direction_reports"]) == {
        "joint",
        "sensing_only",
        "capture_only",
    }
    assert all(row["passed"] for row in result["direction_reports"].values())
    assert result["stage_path_gradient_fraction"] >= 0.01
    assert result["forward_parity_absolute"] <= 1e-6
    assert result["test_accessed"] is False
