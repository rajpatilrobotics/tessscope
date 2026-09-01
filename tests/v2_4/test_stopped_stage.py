"""Integrity checks for matched v2.4 stopped-stage controls."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "stopped-stage-matched.json"
)


def test_stopped_stage_controls_match_frozen_exact_budgets() -> None:
    report = json.loads(ARTIFACT.read_text())
    assert report["status"] == "complete_matched_stopped_stage"
    assert report["test_accessed"] is False
    assert report["hard_labels_accessed"] is False
    assert len(report["pairs"]) == 3
    for pair in report["pairs"]:
        exact = pair["exact"]
        stopped = pair["stopped"]
        assert stopped["profile"] == exact["profile"]
        assert stopped["start_name"] == exact["start_name"]
        assert stopped["steps"] == exact["step"]
        assert len(stopped["trace"]) == exact["step"]
        assert stopped["gradient_mode"] == "stop_stage"


def test_stopped_stage_forward_parity_is_exact() -> None:
    report = json.loads(ARTIFACT.read_text())
    assert report["maximum_forward_parity_absolute"] == 0.0
    assert report["forward_parity_tolerance"] == 1e-6
    for pair in report["pairs"]:
        stopped = pair["stopped"]
        assert stopped["maximum_forward_parity_absolute"] == 0.0
        assert stopped["stopped_validation"] == stopped["exact_forward_validation"]
