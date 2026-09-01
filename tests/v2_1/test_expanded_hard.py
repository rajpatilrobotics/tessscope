"""Tests for expanded B7 hard-validation pairing."""

import pytest

from tessscope.v2_1.evaluation import paired_hard_pq_bootstrap


def metric_row(design: str, well: str, depth: float, pq: float) -> dict:
    return {
        "design": design,
        "well": well,
        "field_id": f"{well}_s1",
        "depth_um": depth,
        "hard_dense_patch": True,
        "panoptic_quality": pq,
    }


def test_paired_bootstrap_groups_depths_by_well() -> None:
    rows = [
        metric_row("candidate", "a01", -2.0, 0.6),
        metric_row("candidate", "a01", 2.0, 0.7),
        metric_row("candidate", "b01", -2.0, 0.5),
        metric_row("candidate", "b01", 2.0, 0.6),
        metric_row("reference", "a01", -2.0, 0.5),
        metric_row("reference", "a01", 2.0, 0.6),
        metric_row("reference", "b01", -2.0, 0.4),
        metric_row("reference", "b01", 2.0, 0.5),
    ]
    result = paired_hard_pq_bootstrap(rows, "candidate", "reference")
    assert result["source_count"] == 2
    assert result["mean_difference"] == pytest.approx(0.1)


def test_paired_bootstrap_rejects_missing_reference_row() -> None:
    rows = [
        metric_row("candidate", "a01", -2.0, 0.6),
        metric_row("reference", "a01", 2.0, 0.5),
    ]
    with pytest.raises(ValueError, match="not paired"):
        paired_hard_pq_bootstrap(rows, "candidate", "reference")
