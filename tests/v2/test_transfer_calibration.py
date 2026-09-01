"""OTF transfer calibration tests."""

from __future__ import annotations

import numpy as np
import pytest

from tessscope.v2.calibration.transfer import (
    fit_transfer_amplitude,
    radial_otf_log_power,
    transfer_ratios,
)


def test_otf_curve_is_zero_for_a_delta_psf() -> None:
    psf = np.zeros((1, 9, 9))
    psf[0, 4, 4] = 1.0

    curve = radial_otf_log_power(psf, output_size=32, radial_bins=4)

    assert np.allclose(curve, 0.0)


def test_transfer_ratios_zero_the_focus_plane() -> None:
    curves = np.arange(21, dtype=np.float64).reshape(7, 3)
    ratios = transfer_ratios(curves)

    assert np.array_equal(ratios[3], np.zeros(3))
    assert np.array_equal(ratios[4], np.full(3, 3.0))


def test_transfer_fit_recovers_positive_amplitude() -> None:
    simulated = np.asarray([-2.0, -1.0, 1.0, 2.0])
    observed = 1.75 * simulated

    fit = fit_transfer_amplitude(observed, simulated)

    assert fit.amplitude == pytest.approx(1.75)
    assert fit.correlation == pytest.approx(1.0)
    assert fit.root_mean_squared_error == pytest.approx(0.0)
