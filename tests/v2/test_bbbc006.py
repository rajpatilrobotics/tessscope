"""BBBC006 filename, geometry, and split-hygiene tests."""

from __future__ import annotations

import pytest

from tessscope.v2.data.bbbc006 import (
    DEPTHS_UM,
    Z_PLANES,
    all_wells,
    build_well_splits,
    parse_image_filename,
    parse_label_filename,
    records_for_split,
    validate_well_splits,
)


def test_bbbc006_depth_mapping_is_centered_on_z16() -> None:
    assert Z_PLANES == (13, 14, 15, 16, 17, 18, 19)
    assert DEPTHS_UM == (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)


def test_parse_official_image_and_label_names() -> None:
    image = parse_image_filename(
        "mcf-z-stacks-03212011_a01_s2_w1a63ef720-8648-425b-a84d-dcf2f6bb0044.tif"
    )
    assert (image.well, image.site, image.channel) == ("a01", 2, 1)
    assert parse_label_filename("mcf-z-stacks-03212011_p24_s1.png") == ("p24", 1)


def test_invalid_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        parse_image_filename("a01_s1_w1.tif")
    with pytest.raises(ValueError):
        parse_label_filename("a01_s1.png")


def test_well_split_is_complete_disjoint_and_row_stratified() -> None:
    splits = build_well_splits()
    validate_well_splits(splits)
    assert len(all_wells()) == 384
    assert {split: len(wells) for split, wells in splits.items()} == {
        "training": 288,
        "validation": 48,
        "test": 48,
    }


def test_test_records_require_explicit_locked_access() -> None:
    with pytest.raises(PermissionError):
        records_for_split("test", require_files=False)
    records = records_for_split("test", allow_test=True, require_files=False)
    assert len(records) == 96
    assert len({record.well for record in records}) == 48
