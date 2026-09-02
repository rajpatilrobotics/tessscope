"""Integrity checks for the final v2.5 sealed-test decision."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLOCK = PROJECT_ROOT / "configs" / "v2_5" / "pretest-block.json"


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()


def test_v2_5_pretest_block_matches_frozen_evidence() -> None:
    block = json.loads(BLOCK.read_text())
    paths = {
        "contract": "configs/v2_5/contract.yaml",
        "source_manifest": "configs/v2_5/source-manifest.json",
        "b11_closed_loop_derivative": (
            "artifacts/runs/v2_5/gates/b11-closed-loop-derivative.json"
        ),
        "b11_constrained": "artifacts/runs/v2_5/optimization/b11-constrained.json",
        "b7_constrained": "artifacts/runs/v2_5/optimization/b7-constrained.json",
    }
    assert block["evidence_sha256"] == {
        name: _sha256(path) for name, path in paths.items()
    }


def test_v2_5_pretest_block_keeps_test_sealed() -> None:
    block = json.loads(BLOCK.read_text())
    assert block["status"] == "failed_soft_promotion_test_sealed"
    assert block["test_accessed"] is False
    assert block["locked_test_run_authorized"] is False
    assert block["b11_optimization"]["selected"] == 0
    assert block["b7_continuity"]["selected"] == 0
    assert "locked_test" in block["conditional_work_not_run"]
