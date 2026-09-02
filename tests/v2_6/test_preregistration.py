"""Integrity checks for the v2.6 diagnosis-first preregistration."""

import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "configs" / "v2_6" / "contract.yaml"
MANIFEST = PROJECT_ROOT / "configs" / "v2_6" / "source-manifest.json"


def test_v2_6_sources_match_frozen_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for source in manifest["sources"].values():
        actual = hashlib.sha256((PROJECT_ROOT / source["path"]).read_bytes()).hexdigest()
        assert actual == source["sha256"]
    assert manifest["source_commit"] == "09b9eb2"
    assert manifest["test_accessed"] is False


def test_v2_6_contract_freezes_diagnostic_and_route_decision() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["status"] == "preregistered_before_v2_6_diagnostic_generation"
    assert contract["frozen_systems"]["candidate"] == (
        "v2_3-exact-balanced-segmentation_only-step-14"
    )
    assert contract["frozen_systems"]["baseline"] == "v2_2-piecewise-028"
    assert contract["targets"]["minimum_corrected_pq_gain_over_matched_baseline"] == 0.005
    assert contract["targets"]["minimum_positive_well_fraction"] == 0.60
    assert contract["targets"]["primary_focus_mae_um"] == 1.0
    assert contract["diagnostic"]["exposure_audit"][
        "total_expected_photons_two_exposures"
    ] == 400.0
    assert contract["joint_route_if_activated"]["feasibility_restoration_first"] is True
    assert contract["test_policy"]["access_runs"] == 1
    assert contract["test_accessed"] is False
