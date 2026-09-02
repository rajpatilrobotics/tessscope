"""End-to-end checks for every displayed claim, sample, and output hash."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessscope.demo.evidence import sha256_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = PROJECT_ROOT / "outputs" / "demo" / "traceability-manifest.json"


def _pointer(document: object, pointer: str) -> object:
    value = document
    for raw_token in pointer.removeprefix("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def test_traceability_inventory_hashes_every_output() -> None:
    trace = json.loads(TRACE_PATH.read_text())
    assert trace["status"] == "complete_end_to_end_validation_traceability"
    assert trace["test_accessed"] is False
    assert trace["scope"] == {
        "allowed_splits": ["training", "validation"],
        "demo_mode": "cached_replay_not_live_inference",
        "deployment": "not_performed",
        "evaluation_split": "validation",
        "forbidden_split": "test",
    }
    assert len(trace["output_inventory"]) == 40
    for path_text, output in trace["output_inventory"].items():
        path = PROJECT_ROOT / path_text
        assert output["sha256"] == sha256_path(path)
        assert output["size_bytes"] == path.stat().st_size


def test_every_claim_pointer_resolves_to_recorded_raw_value() -> None:
    trace = json.loads(TRACE_PATH.read_text())
    expected_ids = {
        "exact_closed_loop_causal_gain",
        "exact_closed_loop_improves_own_first_frame",
        "exact_focus_accuracy",
        "exact_piecewise_comparison",
        "exact_first_frame_segmentation_gate",
        "v2_5_followup",
        "v2_6_followup",
    }
    assert {claim["id"] for claim in trace["claim_trace"]} == expected_ids
    for claim in trace["claim_trace"]:
        source_path = PROJECT_ROOT / claim["source"]
        assert claim["source_sha256"] == sha256_path(source_path)
        source = json.loads(source_path.read_text())
        for value in claim["values"].values():
            assert value["raw"] == _pointer(source, value["pointer"])

    protected = next(
        claim for claim in trace["claim_trace"] if claim["id"] == "exact_closed_loop_causal_gain"
    )
    assert protected["values"]["estimate"]["raw"] == pytest.approx(0.006258840610359453)
    assert protected["values"]["ci_lower_95"]["raw"] > 0.0
    piecewise = next(
        claim for claim in trace["claim_trace"] if claim["id"] == "exact_piecewise_comparison"
    )
    assert piecewise["values"]["ci_lower_95"]["raw"] < 0.0
    assert piecewise["values"]["ci_upper_95"]["raw"] > 0.0


def test_representative_sample_and_display_transform_are_frozen() -> None:
    trace = json.loads(TRACE_PATH.read_text())
    sample = trace["representative_sample"]
    assert sample["population"] == {
        "depths_um": [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0],
        "field_count": 27,
        "frame_count": 162,
    }
    assert sample["selected_patch"]["field_id"] == "n21_s1"
    assert sample["selected_patch"]["origin_yx"] == [0, 220]
    assert sample["selected_display_depth"]["depth_um"] == -2.0
    assert trace["replay_arrays"]["display_transform"]["per_image_normalization"] is False
    assert trace["replay_arrays"]["display_transform"]["scale"] == pytest.approx(
        11.582016617246811
    )
    assert len(trace["replay_arrays"]["patch_sources"]) == 3
    assert set(trace["replay_arrays"]["design_sources"]) == {
        "clear",
        "exact",
        "stopped",
        "piecewise",
    }


def test_every_depth_replay_number_maps_to_one_frozen_row() -> None:
    trace = json.loads(TRACE_PATH.read_text())
    frames = trace["displayed_frame_values"]
    assert len(frames) == 7
    for frame in frames:
        assert len(frame["systems"]) == 3
        for system in frame["systems"]:
            source_path = PROJECT_ROOT / system["source"]
            assert system["source_sha256"] == sha256_path(source_path)
            source = json.loads(source_path.read_text())
            before = _pointer(source, system["before_pointer"])
            assert before["panoptic_quality"] == system["before_pq"]
            assert before["predicted_depth_um"] == system["predicted_depth_um"]
            if system["corrected_pointer"] is not None:
                corrected = _pointer(source, system["corrected_pointer"])
                assert corrected["after_pq"] == system["after_pq"]
                assert corrected["residual_depth_um"] == system["residual_depth_um"]


def test_population_and_optical_displays_record_derivations() -> None:
    trace = json.loads(TRACE_PATH.read_text())
    population = trace["displayed_population_curves"]
    assert "resamples wells" in population["derivation"]
    assert set(population["pq_curves"]) == {
        "exact_before",
        "exact_after",
        "stopped_after",
        "piecewise_after",
    }
    assert set(population["focus_curves"]) == {"exact", "stopped", "piecewise"}
    optical = trace["displayed_optical_values"]
    assert set(optical["array_entries"]) == {
        "phase_parameters",
        "pupil_phase_radians",
        "pupil_mask",
        "psf_sensor",
    }
    assert optical["render"]["limits"] == [-5.0, 0.0]
    architecture = trace["displayed_architecture_values"]
    assert architecture["passed"] is True
    assert architecture["forward_parity_absolute"] == 0.0


def test_static_demo_facts_derive_from_replay_metadata() -> None:
    trace = json.loads(TRACE_PATH.read_text())
    facts = trace["demo_static_facts"]
    source_path = PROJECT_ROOT / facts["source"]
    source = json.loads(source_path.read_text())
    assert facts["source_sha256"] == sha256_path(source_path)
    assert facts["source_pack_size_bytes"] == source["archive"]["size_bytes"]
    assert facts["source_pack_size_display_mib"] == 12.4
    assert facts["depth_count"] == len(source["depths_um"]) == 7
    assert facts["system_count"] == len(source["system_order"]) == 4


def test_traceability_manifest_is_byte_stable() -> None:
    assert sha256_path(TRACE_PATH) == (
        "8bb86fed42466bc71f2ef7e5ec358c827a568ff8c2dc468c43dd98be4ea8fd9a"
    )
