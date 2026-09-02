"""Integrity checks for the preregistered evidence-first demo contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PATH = PROJECT_ROOT / "configs" / "demo" / "claim-matrix.yaml"
VISUAL_PATH = PROJECT_ROOT / "configs" / "demo" / "visual-contract.yaml"
MANIFEST_PATH = PROJECT_ROOT / "configs" / "demo" / "source-manifest.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _pointer(document: object, pointer: str) -> object:
    value = document
    for raw_token in pointer.removeprefix("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def test_source_manifest_hashes_are_frozen_and_valid() -> None:
    manifest = _load_json(MANIFEST_PATH)
    assert manifest["status"] == "frozen_before_example_selection_or_rendering"
    assert manifest["data_boundary"] == {
        "allowed_splits": ["training", "validation"],
        "qualitative_split": "validation",
        "forbidden_split": "test",
        "test_accessed": False,
    }
    for source in manifest["sources"].values():
        payload = (PROJECT_ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["sha256"]


def test_claim_pointers_resolve_and_protected_values_match() -> None:
    claims = yaml.safe_load(CLAIMS_PATH.read_text())
    protected = claims["protected_claim"]
    source = _load_json(PROJECT_ROOT / protected["source"])
    resolved = {
        name: _pointer(source, pointer) for name, pointer in protected["pointers"].items()
    }
    assert resolved["estimate"] == pytest.approx(0.006258840610359453)
    assert resolved["ci_lower_95"] == pytest.approx(0.0006703181908061804)
    assert resolved["ci_upper_95"] == pytest.approx(0.011547458540670894)
    assert resolved["source_count"] == 27
    assert resolved["bootstrap_replicates"] == 2000

    for claim in claims["claims"]:
        document = _load_json(PROJECT_ROOT / claim["source"])
        for pointer in claim.get("pointers", {}).values():
            _pointer(document, pointer)


def test_contract_preserves_required_limitations_and_render_rules() -> None:
    claims = yaml.safe_load(CLAIMS_PATH.read_text())
    limitations = {row["id"]: row["required_text"] for row in claims["limitations"]}
    assert "not percent accuracy" in limitations["pq_not_accuracy"]
    assert "validation-only" in limitations["validation_scope"]
    assert "remains sealed" in limitations["test_seal"]
    assert "No physical microscope" in limitations["hardware_scope"]
    assert "cached evidence" in limitations["replay_scope"]

    visual = yaml.safe_load(VISUAL_PATH.read_text())
    assert visual["status"] == "preregistered_before_example_selection_or_rendering"
    assert visual["representative_selection"]["selection"] == "minimum_distance"
    assert visual["representative_selection"]["prohibition"].startswith("rendered appearance")
    assert visual["display_transform"]["per_image_normalization"] is False
    assert visual["crop_and_geometry"]["scale_bar"] == "omitted"
    assert visual["demo"]["mode"] == "local_static_cached_replay"
    assert visual["demo"]["locked_test_access"] is False


def test_frozen_evidence_sources_report_no_locked_test_access() -> None:
    manifest = _load_json(MANIFEST_PATH)
    for key in (
        "v2_4_hard_validation",
        "v2_2_piecewise_hard_validation",
        "v2_5_pretest_block",
        "v2_6_status",
        "v2_6_pretest_block",
    ):
        source = _load_json(PROJECT_ROOT / manifest["sources"][key]["path"])
        assert source["test_accessed"] is False
