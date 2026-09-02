"""Frozen evidence checks for the terminal v2.6 two-mask result."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "two-mask"
    / "matched-baseline-screen.json"
)
OPTIMIZATION = BASELINES.with_name("exact-optimization.json")
BLOCK = PROJECT_ROOT / "configs" / "v2_6" / "pretest-block.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_76_matched_pairs_were_screened_without_promotion() -> None:
    result = json.loads(BASELINES.read_text())
    assert len(result["development_rows"]) == 76
    assert len(
        {
            (row["sensing_name"], row["capture_name"])
            for row in result["development_rows"]
        }
    ) == 76
    assert not any(row["eligible"] for row in result["development_rows"])
    assert result["selected_for_confirmation"] == []
    assert result["best_confirmation_baseline_name"] is None
    assert result["test_accessed"] is False


def test_complete_exact_matrix_freezes_zero_candidates() -> None:
    result = json.loads(OPTIMIZATION.read_text())
    assert result["status"] == "complete_training_only_negative"
    assert len(result["restorations"]) == 4
    assert not any(row["feasible"] for row in result["restorations"])
    assert len(result["endpoints"]) == 4
    assert sum(len(row["trace"]) for row in result["endpoints"]) == 72
    assert not any(
        row["eligible_without_matched_baseline"]
        for row in result["development_selection_rows"]
    )
    assert result["selected_for_confirmation"] == []
    assert result["selected_for_hard_validation"] == []
    assert result["test_accessed"] is False


def test_pretest_block_hashes_final_evidence_and_keeps_test_sealed() -> None:
    block = json.loads(BLOCK.read_text())
    assert block["source_hashes"]["matched_baseline_screen"] == sha256(BASELINES)
    assert block["source_hashes"]["exact_optimization"] == sha256(OPTIMIZATION)
    assert block["locked_test_decision"] == "do_not_access"
    assert block["test_accessed"] is False
