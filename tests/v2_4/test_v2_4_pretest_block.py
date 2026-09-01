"""Integrity checks for the frozen negative v2.4 hard decision."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCK = PROJECT_ROOT / "configs" / "v2_4" / "pretest-block.json"
HARD = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)


def test_v2_4_block_records_negative_and_keeps_test_sealed() -> None:
    block = json.loads(BLOCK.read_text())
    assert block["status"] == "failed_expanded_hard_promotion_test_sealed"
    assert block["test_accessed"] is False
    assert block["locked_test_run_authorized"] is False
    assert block["checkpoint_audit"]["soft_eligible"] == 90
    assert block["hard_protocol"]["hard_passed_count"] == 0
    assert block["best_corrected_exact"]["passed"] is False


def test_v2_4_hard_artifact_is_complete_and_has_no_promotion() -> None:
    report = json.loads(HARD.read_text())
    assert report["status"] == "complete_expanded_hard_validation_only"
    assert report["test_accessed"] is False
    assert report["coverage"]["evaluated_validation_wells"] == 45
    assert report["hard_definition"]["well_count"] == 45
    assert len(report["hard_definition"]["hard_field_ids"]) == 27
    assert len(report["design_registry"]) == 6
    assert len(report["rows"]) == 1890
    assert len(report["corrected_rows"]) == 972
    assert report["hard_passed_names"] == []
    assert report["selected_for_matched_derivative_free"] is None
    assert all(not gates["passed_all_hard_gates"] for gates in report["gates"].values())


def test_v2_4_evidence_hashes_match() -> None:
    block = json.loads(BLOCK.read_text())
    paths = {
        "contract": "configs/v2_4/contract.yaml",
        "checkpoint_pool": (
            "artifacts/runs/v2_4/preregistration/checkpoint-pool.json"
        ),
        "checkpoint_audit": (
            "artifacts/runs/v2_4/validation/checkpoint-audit.json"
        ),
        "selected_checkpoints": "configs/v2_4/selected-checkpoints.json",
        "stopped_stage": (
            "artifacts/runs/v2_4/validation/stopped-stage-matched.json"
        ),
        "expanded_hard_validation": (
            "artifacts/runs/v2_4/validation/expanded-hard-validation.json"
        ),
        "closed_loop_derivative": (
            "artifacts/runs/v2_3/gates/closed-loop-derivative.json"
        ),
    }
    for key, relative_path in paths.items():
        digest = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == block["evidence_sha256"][key]
