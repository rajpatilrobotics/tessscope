"""Convention, constraint, and PSF checks for the v2.1 B11 pupil."""

import jax.numpy as jnp
import numpy as np
import pytest

from tessscope.v2_1.optics.model import (
    phase_coefficients_b11,
    psf_sensor_b11,
    psf_support_energy_fraction_b11,
    unconstrained_from_coefficients_b11,
    zernike_basis_b11,
)


def test_b11_sampled_basis_is_nearly_orthonormal_on_unit_disk() -> None:
    axis = np.linspace(-1.0, 1.0, 401, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    grid = np.stack([yy, xx], axis=-1)
    inside = xx * xx + yy * yy <= 1.0
    basis = np.asarray(zernike_basis_b11(jnp.asarray(grid), 1.0))[:, inside]
    gram = basis @ basis.T / basis.shape[1]
    assert np.diag(gram) == pytest.approx(np.ones(11), abs=0.012)
    assert np.max(np.abs(gram - np.diag(np.diag(gram)))) < 0.012
    assert np.max(np.abs(np.mean(basis[7:], axis=1))) < 0.006


def test_b11_open_ball_round_trip_and_shape_guard() -> None:
    coefficients = np.asarray(
        [0.2, -0.1, 0.3, 0.0, 0.2, -0.2, 0.8, 0.1, -0.1, 0.2, -0.2]
    )
    parameters = unconstrained_from_coefficients_b11(coefficients)
    recovered = np.asarray(phase_coefficients_b11(jnp.asarray(parameters)))
    assert recovered == pytest.approx(coefficients, abs=1e-6)
    assert np.linalg.norm(recovered) < 2.5
    with pytest.raises(ValueError, match="11"):
        phase_coefficients_b11(jnp.zeros(7))


def test_b11_psf_energy_and_support() -> None:
    coefficients = np.zeros(11)
    coefficients[7:] = [0.3, -0.2, 0.2, -0.3]
    parameters = unconstrained_from_coefficients_b11(coefficients)
    psf = np.asarray(psf_sensor_b11(jnp.asarray(parameters), jnp.asarray(6.0)))
    support = float(
        psf_support_energy_fraction_b11(
            jnp.asarray(parameters), jnp.asarray(6.0)
        )
    )
    assert psf.shape == (96, 96)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)
    assert support >= 0.995
