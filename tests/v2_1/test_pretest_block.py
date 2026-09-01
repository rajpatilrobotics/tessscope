"""Integrity checks for the frozen v2.1 negative pre-test checkpoint."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v2_1_pretest_block_keeps_locked_test_sealed() -> None:
    block = json.loads(
        (PROJECT_ROOT / "configs/v2_1/pretest-block.json").read_text()
    )
    assert block["status"] == "complete_negative_pretest_test_sealed"
    assert block["test_accessed"] is False
    assert block["locked_test_run_authorized"] is False
    assert block["validation_coverage"]["evaluated_wells"] == 45


def test_v2_1_pretest_evidence_hashes_match() -> None:
    block = json.loads(
        (PROJECT_ROOT / "configs/v2_1/pretest-block.json").read_text()
    )
    paths = {
        "b7_basis_diagnostics": "artifacts/runs/v2_1/gates/b7-basis-diagnostics.json",
        "b7_three_tesseract_derivative": (
            "artifacts/runs/v2_1/gates/b7-three-tesseract-derivative.json"
        ),
        "b7_separate_baselines": "artifacts/runs/v2_1/optimization/b7-separate-baselines.json",
        "b7_projected_candidates": (
            "artifacts/runs/v2_1/optimization/b7-projected-gradient-candidates.json"
        ),
        "b7_expanded_hard_validation": (
            "artifacts/runs/v2_1/validation/expanded-hard-validation.json"
        ),
        "b11_basis_diagnostics": "artifacts/runs/v2_1/gates/b11-basis-diagnostics.json",
        "b11_three_tesseract_derivative": (
            "artifacts/runs/v2_1/gates/b11-three-tesseract-derivative.json"
        ),
        "b11_separate_baselines": "artifacts/runs/v2_1/optimization/b11-separate-baselines.json",
        "b11_projected_candidates": (
            "artifacts/runs/v2_1/optimization/b11-projected-gradient-candidates.json"
        ),
        "b11_expanded_hard_validation": (
            "artifacts/runs/v2_1/validation/b11-expanded-hard-validation.json"
        ),
    }
    for key, relative_path in paths.items():
        digest = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == block["evidence_sha256"][key]
