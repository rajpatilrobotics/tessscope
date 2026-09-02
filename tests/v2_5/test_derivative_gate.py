"""Frozen integrity assertions for the v2.5 B11 full-loop derivative gate."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_5"
    / "gates"
    / "b11-closed-loop-derivative.json"
)
MANIFEST = PROJECT_ROOT / "configs" / "v2_5" / "source-manifest.json"


def test_v2_5_b11_full_loop_derivative_gate_passed_without_test_access() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    source = next(
        row for row in manifest["starts"] if row["name"] == "b11_segmentation_only"
    )
    assert evidence["parameter_sha256"] == source["parameter_sha256"]
    assert evidence["overall_median_relative_error"] < 0.01
    assert evidence["overall_cosine_agreement"] > 0.99
    assert evidence["stage_path_gradient_fraction"] >= 0.01
    assert evidence["forward_parity_absolute"] <= 1e-6
    assert evidence["passed"] is True
    assert evidence["passed_v2_5"] is True
    assert evidence["test_accessed"] is False
