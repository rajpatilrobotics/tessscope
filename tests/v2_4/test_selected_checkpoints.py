"""Integrity tests for the frozen v2.4 soft checkpoint selection."""

import hashlib
import json
from pathlib import Path

from tessscope.v2_4.checkpoints import select_checkpoints

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION = PROJECT_ROOT / "configs" / "v2_4" / "selected-checkpoints.json"
AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "checkpoint-audit.json"
)


def parameter_sha256(parameters: list[float]) -> str:
    payload = json.dumps(parameters, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def test_frozen_selection_reproduces_from_all_soft_rows() -> None:
    audit = json.loads(AUDIT.read_text())
    frozen = json.loads(SELECTION.read_text())
    reproduced = select_checkpoints(audit["rows"])
    assert reproduced == audit["selection"]
    assert [row["name"] for row in frozen["selected"]] == reproduced[
        "selected_names"
    ]
    assert frozen["pool_counts"] == {
        "checkpoint_records": 270,
        "unique_parameter_hashes": 269,
        "soft_eligible": 90,
        "soft_nondominated": 57,
        "selected": 3,
    }


def test_frozen_selection_hashes_parameters_and_keeps_test_sealed() -> None:
    frozen = json.loads(SELECTION.read_text())
    assert frozen["status"] == (
        "soft_selection_frozen_before_stopped_or_hard_validation"
    )
    assert frozen["test_accessed"] is False
    assert frozen["hard_labels_accessed"] is False
    assert frozen["first_segmentation_maximum"] == 1.098270310640335
    assert len({row["source_run"] for row in frozen["selected"]}) == 3
    assert len({row["parameter_sha256"] for row in frozen["selected"]}) == 3
    for row in frozen["selected"]:
        assert parameter_sha256(row["parameters"]) == row["parameter_sha256"]


def test_frozen_selection_evidence_hashes_match() -> None:
    frozen = json.loads(SELECTION.read_text())
    paths = {
        "contract": "configs/v2_4/contract.yaml",
        "checkpoint_pool": (
            "artifacts/runs/v2_4/preregistration/checkpoint-pool.json"
        ),
        "checkpoint_audit": (
            "artifacts/runs/v2_4/validation/checkpoint-audit.json"
        ),
    }
    for key, relative_path in paths.items():
        digest = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == frozen["evidence_sha256"][key]
