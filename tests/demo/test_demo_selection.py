"""Tests for the frozen representative validation selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessscope.demo.evidence import (
    DEPTH_FEATURES,
    EXACT_NAME,
    FIELD_FEATURES,
    PIECEWISE_NAME,
    STOPPED_NAME,
    robust_rank,
    select_representative,
    sha256_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_4 = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)
PIECEWISE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "validation"
    / "piecewise-hard-frontier.json"
)
SELECTION = PROJECT_ROOT / "artifacts" / "runs" / "demo" / "representative-selection.json"


def test_robust_rank_uses_hash_only_after_distance() -> None:
    rows = [
        {"id": "far", "score": 10.0},
        {"id": "middle-b", "score": 2.0},
        {"id": "middle-a", "score": 2.0},
        {"id": "low", "score": 0.0},
    ]
    ranked, statistics = robust_rank(
        rows, identifier_key="id", feature_names=("score",), seed="test"
    )
    assert ranked[0]["representative_distance"] == ranked[1]["representative_distance"]
    assert [row["representative_distance"] for row in ranked] == sorted(
        row["representative_distance"] for row in ranked
    )
    assert statistics["score"]["median"] == pytest.approx(2.0)


def test_selection_recomputes_exactly_from_frozen_metric_rows() -> None:
    frozen = json.loads(SELECTION.read_text())
    recomputed = select_representative(
        json.loads(V2_4.read_text()), json.loads(PIECEWISE.read_text())
    )
    for key in (
        "selection_rule",
        "population",
        "field_feature_statistics",
        "field_ranking",
        "selected_field",
        "selected_depth_feature_statistics",
        "selected_field_depth_ranking",
        "selected_display_depth",
    ):
        assert frozen[key] == recomputed[key]

    assert frozen["population"] == {
        "depths_um": [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0],
        "field_count": 27,
        "frame_count": 162,
    }
    assert frozen["selected_patch"] == {
        "field_id": "n21_s1",
        "origin_yx": [0, 220],
        "shape_px": [256, 256],
        "site": 1,
        "valid_instance_count": 4,
        "well": "n21",
    }
    assert frozen["selected_display_depth"]["depth_um"] == -2.0


def test_selection_uses_only_registered_systems_and_validation() -> None:
    frozen = json.loads(SELECTION.read_text())
    assert frozen["status"] == "selected_before_visual_rendering"
    assert frozen["split"] == "validation"
    assert frozen["test_accessed"] is False
    assert frozen["selection_rule"]["field_features"] == list(FIELD_FEATURES)
    assert frozen["selection_rule"]["display_depth_features"] == list(DEPTH_FEATURES)

    v2_4 = json.loads(V2_4.read_text())
    exact_rows = [row for row in v2_4["corrected_rows"] if row["design"] == EXACT_NAME]
    stopped_rows = [row for row in v2_4["corrected_rows"] if row["design"] == STOPPED_NAME]
    piecewise = json.loads(PIECEWISE.read_text())
    piecewise_rows = [
        row for row in piecewise["corrected_rows"] if row["design"] == PIECEWISE_NAME
    ]
    assert len(exact_rows) == len(stopped_rows) == len(piecewise_rows) == 162
    assert sha256_path(SELECTION) == (
        "52be9bbf261c033649d4b502384da110d93f5525ffbab3fef5e2e7efabd0d4d3"
    )
