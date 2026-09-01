"""Integrity checks for the frozen negative v2.3 closed-loop decision."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v2_3_block_keeps_test_sealed_and_records_failure() -> None:
    block = json.loads(
        (PROJECT_ROOT / "configs/v2_3/pretest-block.json").read_text()
    )
    assert block["status"] == "failed_soft_promotion_test_sealed"
    assert block["test_accessed"] is False
    assert block["locked_test_run_authorized"] is False
    assert block["optimization"]["exact_runs"] == 9
    assert block["optimization"]["completed_exact_steps"] == 270
    assert block["optimization"]["soft_eligible_count"] == 0
    assert block["soft_gate"]["closest_endpoint"]["passed"] is False
    assert block["derivative_gate"]["passed"] is True


def test_v2_3_evidence_hashes_match() -> None:
    block = json.loads(
        (PROJECT_ROOT / "configs/v2_3/pretest-block.json").read_text()
    )
    paths = {
        "contract": "configs/v2_3/contract.yaml",
        "v2_2_pretest_block": "configs/v2_2/pretest-block.json",
        "closed_loop_derivative": (
            "artifacts/runs/v2_3/gates/closed-loop-derivative.json"
        ),
        "closed_loop_matrix": (
            "artifacts/runs/v2_3/optimization/closed-loop-matrix.json"
        ),
    }
    for key, relative_path in paths.items():
        digest = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == block["evidence_sha256"][key]


def test_v2_3_matrix_has_no_soft_eligible_endpoint() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/runs/v2_3/optimization/closed-loop-matrix.json"
        ).read_text()
    )
    maximum = report["first_segmentation_maximum"]
    assert report["status"] == "complete_training_validation_negative"
    assert report["test_accessed"] is False
    assert len(report["exact_runs"]) == 9
    assert report["soft_eligible_exact_names"] == []
    assert report["selected_for_hard_validation"] == []
    assert report["stopped_stage_ablation"] is None
    for run in report["exact_runs"]:
        assert run["steps"] == 30
        assert len(run["trace"]) == 30
        assert run["validation"]["residual_mae_um"] < run["start_validation"][
            "residual_mae_um"
        ]
        assert run["validation"]["first_segmentation_loss"] > maximum
