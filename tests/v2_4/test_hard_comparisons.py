"""Pure tests for v2.4 hard paired comparisons and stability gates."""

import pytest

from tessscope.v2_4.hard import paired_corrected_pq_bootstrap, well_stability


def corrected(design: str, well: str, depth: float, value: float) -> dict:
    return {
        "design": design,
        "well": well,
        "field_id": f"{well}_field",
        "original_depth_um": depth,
        "after_pq": value,
    }


def test_corrected_bootstrap_is_candidate_minus_reference() -> None:
    candidate = [
        corrected("candidate", well, depth, 0.7)
        for well in ("a", "b", "c")
        for depth in (-2.0, 2.0)
    ]
    reference = [
        corrected("reference", well, depth, 0.5)
        for well in ("a", "b", "c")
        for depth in (-2.0, 2.0)
    ]
    result = paired_corrected_pq_bootstrap(
        candidate, reference, "candidate", "reference"
    )
    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["ci_lower_95"] == pytest.approx(0.2)


def test_stability_rejects_single_well_dependence() -> None:
    candidate = [
        corrected("candidate", "a", -2.0, 1.0),
        corrected("candidate", "b", -2.0, 0.4),
        corrected("candidate", "c", -2.0, 0.4),
    ]
    reference = [
        corrected("reference", well, -2.0, 0.5) for well in ("a", "b", "c")
    ]
    result = well_stability(
        candidate,
        reference,
        candidate_name="candidate",
        reference_name="reference",
        depth_key="original_depth_um",
        metric_key="after_pq",
    )
    assert result["positive_well_fraction"] == pytest.approx(1.0 / 3.0)
    assert result["minimum_leave_one_well_out_mean_difference"] < 0.0
    assert result["passed"] is False


def test_stability_accepts_distributed_positive_effect() -> None:
    candidate = [
        corrected("candidate", well, -2.0, value)
        for well, value in (("a", 0.7), ("b", 0.6), ("c", 0.45))
    ]
    reference = [
        corrected("reference", well, -2.0, 0.5) for well in ("a", "b", "c")
    ]
    result = well_stability(
        candidate,
        reference,
        candidate_name="candidate",
        reference_name="reference",
        depth_key="original_depth_um",
        metric_key="after_pq",
    )
    assert result["positive_well_fraction"] == pytest.approx(2.0 / 3.0)
    assert result["minimum_leave_one_well_out_mean_difference"] > 0.0
    assert result["passed"] is True
