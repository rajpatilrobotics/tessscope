"""Tests for validation-only near-miss audit helpers."""

import pytest

from tessscope.v2_1.audit import paired_comparison, summarize_hard_rows


def _row(field: str, depth: float, prediction: float, pq: float) -> dict:
    return {
        "well": field[0],
        "field_id": field,
        "depth_um": depth,
        "predicted_depth_um": prediction,
        "hard_dense_patch": True,
        "panoptic_quality": pq,
        "segmentation_quality": pq + 0.1,
        "recognition_quality": pq - 0.1,
        "foreground_dice": pq + 0.2,
        "absolute_count_error": 1.0 - pq,
    }


def test_summarize_hard_rows_keeps_signed_focus_and_pq() -> None:
    rows = [
        _row("a_s1", -2.0, -1.5, 0.4),
        _row("a_s1", 0.0, 0.1, 0.5),
        _row("a_s1", 2.0, 1.5, 0.6),
    ]
    summary = summarize_hard_rows(rows)
    assert summary["hard_dense_off_focus"]["panoptic_quality"] == pytest.approx(0.5)
    assert summary["focus"]["signed_direction_accuracy"] == 1.0
    assert summary["focus"]["mae_um"] == pytest.approx(1.1 / 3.0)


def test_paired_comparison_reports_candidate_minus_reference() -> None:
    reference = [_row("a_s1", -2.0, -1.0, 0.4), _row("b_s1", 2.0, 1.0, 0.5)]
    candidate = [_row("a_s1", -2.0, -1.5, 0.5), _row("b_s1", 2.0, 1.5, 0.7)]
    rows = paired_comparison(reference, candidate, group_key="well")
    assert rows[0]["delta_panoptic_quality"] == pytest.approx(0.1)
    assert rows[0]["delta_focus_absolute_error_um"] == pytest.approx(-0.5)


def test_paired_comparison_rejects_membership_mismatch() -> None:
    with pytest.raises(ValueError, match="field/depth membership"):
        paired_comparison(
            [_row("a_s1", -2.0, -1.0, 0.4)],
            [_row("a_s1", 2.0, 1.0, 0.4)],
            group_key="well",
        )
