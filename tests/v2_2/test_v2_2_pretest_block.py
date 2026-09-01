"""Integrity checks for the frozen negative v2.2 frontier decision."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v2_2_block_keeps_test_sealed_and_records_failure() -> None:
    block = json.loads(
        (PROJECT_ROOT / "configs/v2_2/pretest-block.json").read_text()
    )
    assert block["status"] == "failed_matched_frontier_test_sealed"
    assert block["test_accessed"] is False
    assert block["locked_test_run_authorized"] is False
    assert block["matched_segmentation"]["passed"] is False
    assert block["matched_focus"]["passed"] is False


def test_v2_2_evidence_hashes_match() -> None:
    block = json.loads(
        (PROJECT_ROOT / "configs/v2_2/pretest-block.json").read_text()
    )
    paths = {
        "contract": "configs/v2_2/contract.yaml",
        "frozen_joint_candidate": "configs/v2_2/frozen-joint-candidate.json",
        "piecewise_soft_frontier": (
            "artifacts/runs/v2_2/screening/piecewise-soft-frontier.json"
        ),
        "piecewise_hard_frontier": (
            "artifacts/runs/v2_2/validation/piecewise-hard-frontier.json"
        ),
    }
    for key, relative_path in paths.items():
        digest = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == block["evidence_sha256"][key]
