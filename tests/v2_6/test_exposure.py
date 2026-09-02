"""Pure tests for the v2.6 fixed-total-photon exposure audit."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from tessscope.v2_6.exposure import (
    exposure_photons,
    select_development_fraction,
    summarize_exposure_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def exposure_rows(fraction: float, before: float, after: float) -> list[dict]:
    return [
        {
            "well": well,
            "first_fraction": fraction,
            "original_depth_um": depth,
            "predicted_depth_um": depth,
            "residual_depth_um": 0.0,
            "before_pq": before,
            "after_pq": after,
            "first_photon_mean": 100.0,
            "second_photon_mean": 100.0,
        }
        for well in ("a01", "e01")
        for depth in (-2.0, 2.0)
    ]


def test_exposure_photons_conserve_total() -> None:
    for fraction in (0.35, 0.425, 0.5, 0.575, 0.65):
        first, second = exposure_photons(400.0, fraction)
        assert np.isclose(first + second, 400.0)
    with pytest.raises(ValueError):
        exposure_photons(400.0, 1.0)


def test_exposure_contract_has_exactly_two_systems() -> None:
    contract = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "v2_6" / "exposure-audit.yaml").read_text()
    )
    assert set(contract["systems"]) == {"candidate", "baseline"}
    assert "piecewise_parameter_source_sha256" in contract["source_evidence"]


def test_exposure_summary_and_selection_preserve_first_frame() -> None:
    rows = [
        *exposure_rows(0.35, 0.49, 0.70),
        *exposure_rows(0.5, 0.50, 0.65),
        *exposure_rows(0.65, 0.51, 0.68),
    ]
    summary = summarize_exposure_rows(rows[:4])
    assert summary["corrected_pq"] == 0.70
    assert summary["focus_mae_um"] == 0.0
    selection = select_development_fraction(rows)
    assert selection["selected_first_fraction"] == 0.65
