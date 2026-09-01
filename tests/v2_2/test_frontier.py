"""Tests for the pre-registered v2.2 B7 piecewise frontier."""

import numpy as np
import pytest

from tessscope.v2_2.evaluation import (
    hard_pareto_names,
    hypervolume_2d,
    paired_focus_mae_bootstrap,
)
from tessscope.v2_2.frontier import (
    PROJECTED_RADIUS_RADIANS,
    generate_piecewise_family,
    nondominated_names,
    project_coefficients,
    select_hard_candidates,
)


def source_coefficients() -> tuple[np.ndarray, np.ndarray]:
    segmentation = np.asarray([0.8, 0.1, 0.0, -0.5, 0.0, 0.1, 0.2])
    focus = np.asarray([-0.6, 0.2, 0.1, 0.0, -0.1, 0.0, 0.1])
    return segmentation, focus


def test_projection_preserves_interior_amplitudes() -> None:
    interior = np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    projected, changed = project_coefficients(interior)
    assert changed is False
    assert projected == pytest.approx(interior)
    outside = np.full(7, 2.0)
    projected, changed = project_coefficients(outside)
    assert changed is True
    assert np.linalg.norm(projected) == pytest.approx(PROJECTED_RADIUS_RADIANS)


def test_family_contains_exact_naive_sum_and_every_registered_alias() -> None:
    segmentation, focus = source_coefficients()
    family = generate_piecewise_family(segmentation, focus)
    naive = [row for row in family if row["includes_original_naive_sum"]]
    assert len(naive) == 1
    assert naive[0]["phase_coefficients"] == pytest.approx(segmentation + focus)
    aliases = [alias for row in family for alias in row["aliases"]]
    assert sum(alias["family"] == "convex_interpolation" for alias in aliases) == 21
    assert sum(alias["family"] == "focus_injection" for alias in aliases) == 31
    assert sum(alias["family"] == "nonnegative_two_weight" for alias in aliases) == 48
    assert sum(alias["family"] == "original_naive_sum" for alias in aliases) == 1
    assert all(row["phase_rms_radians"] <= PROJECTED_RADIUS_RADIANS for row in family)


def test_nondominated_and_guard_band_selection() -> None:
    rows = [
        {
            "name": "a",
            "segmentation_loss": 1.0,
            "focus_mse": 4.0,
            "is_family_anchor": True,
            "includes_original_naive_sum": False,
        },
        {
            "name": "b",
            "segmentation_loss": 1.1,
            "focus_mse": 2.0,
            "is_family_anchor": False,
            "includes_original_naive_sum": False,
        },
        {
            "name": "c",
            "segmentation_loss": 1.102,
            "focus_mse": 2.1,
            "is_family_anchor": False,
            "includes_original_naive_sum": True,
        },
        {
            "name": "d",
            "segmentation_loss": 1.2,
            "focus_mse": 5.0,
            "is_family_anchor": False,
            "includes_original_naive_sum": False,
        },
    ]
    assert nondominated_names(rows) == {"a", "b"}
    selected, reasons = select_hard_candidates(
        rows,
        joint_segmentation_loss=1.1,
        joint_focus_mse=2.0,
    )
    assert {"a", "b", "c"}.issubset(selected)
    assert "soft_envelope_guard_band" in reasons["c"]
    assert "original_naive_sum" in reasons["c"]


def test_hard_pareto_and_hypervolume() -> None:
    summaries = {
        "seg": {"hard_dense_off_focus_pq": 0.5, "mae_um": 1.8},
        "middle": {"hard_dense_off_focus_pq": 0.48, "mae_um": 1.2},
        "dominated": {"hard_dense_off_focus_pq": 0.47, "mae_um": 1.3},
        "focus": {"hard_dense_off_focus_pq": 0.4, "mae_um": 0.8},
    }
    assert hard_pareto_names(summaries) == {"seg", "middle", "focus"}
    assert hypervolume_2d([(1.0, 0.5), (0.5, 1.0)]) == pytest.approx(0.75)


def test_focus_bootstrap_pairs_depths_within_wells() -> None:
    rows = []
    for design, error in (("joint", 0.5), ("piecewise", 0.8)):
        for well in ("a01", "b01"):
            for depth in (-2.0, 0.0, 2.0):
                rows.append(
                    {
                        "design": design,
                        "well": well,
                        "depth_um": depth,
                        "predicted_depth_um": depth + error,
                    }
                )
    result = paired_focus_mae_bootstrap(rows, "joint", "piecewise")
    assert result["source_count"] == 2
    assert result["mean_difference"] == pytest.approx(0.3)
