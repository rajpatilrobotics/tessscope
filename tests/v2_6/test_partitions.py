"""Tests for the frozen v2.6 training-only well partitions."""

import json
from pathlib import Path

from tessscope.v2_6.partitions import build_partitions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-splits.json"
FROZEN = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "v2_6"
    / "training-development-confirmation.json"
)


def test_v2_6_partitions_are_disjoint_training_only_and_balanced() -> None:
    source = json.loads(SOURCE.read_text())
    result = build_partitions(source["wells"]["training"])
    assert {name: len(wells) for name, wells in result.items()} == {
        "optimization": 192,
        "development": 48,
        "confirmation": 48,
    }
    groups = [set(wells) for wells in result.values()]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == set(source["wells"]["training"])
    assert not set.union(*groups) & set(source["wells"]["test"])
    for wells, expected_per_row in zip(groups, (12, 3, 3), strict=True):
        assert all(
            sum(well.startswith(row) for well in wells) == expected_per_row
            for row in "abcdefghijklmnop"
        )


def test_v2_6_frozen_partition_matches_generator() -> None:
    source = json.loads(SOURCE.read_text())
    frozen = json.loads(FROZEN.read_text())
    assert frozen["wells"] == build_partitions(source["wells"]["training"])
    assert frozen["test_wells_accessed"] is False
