"""Integrity tests for the preregistered frozen v2.4 checkpoint pool."""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_3"
    / "optimization"
    / "closed-loop-matrix.json"
)
MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "preregistration"
    / "checkpoint-pool.json"
)


def parameter_sha256(parameters: list[float]) -> str:
    payload = json.dumps(parameters, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def test_v2_4_manifest_matches_every_frozen_source_checkpoint() -> None:
    source = json.loads(SOURCE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    expected = []
    for run in source["exact_runs"]:
        assert len(run["trace"]) == 30
        for row in run["trace"]:
            step = int(row["step"])
            expected.append(
                {
                    "name": f"{run['name']}-step-{step:02d}",
                    "source_run": run["name"],
                    "profile": run["profile"],
                    "start_name": run["start_name"],
                    "step": step,
                    "parameter_sha256": parameter_sha256(row["parameters"]),
                }
            )
    assert manifest["records"] == expected
    assert manifest["source_run_count"] == 9
    assert manifest["checkpoint_records_per_run"] == 30
    assert manifest["checkpoint_record_count"] == 270


def test_v2_4_manifest_freezes_source_and_contains_no_validation_metrics() -> None:
    manifest = json.loads(MANIFEST.read_text())
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest["source_artifact_sha256"] == source_digest
    assert source_digest == (
        "27ca34997f2ff574093c00fee2f06b6987ee902ce98801b1d3ae2a08cecebcaf"
    )
    assert manifest["status"] == "frozen_before_intermediate_validation"
    assert manifest["intermediate_validation_metrics_computed"] is False
    assert manifest["test_accessed"] is False
    assert len(manifest["controls"]) == 3
    assert all(control["selectable"] is False for control in manifest["controls"])
    forbidden = {
        "objective",
        "first_segmentation_loss",
        "final_segmentation_loss",
        "residual_mae_um",
        "eligible",
    }
    assert all(forbidden.isdisjoint(record) for record in manifest["records"])
