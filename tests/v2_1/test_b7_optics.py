"""Convention, constraint, and PSF checks for the v2.1 B7 pupil."""

import jax.numpy as jnp
import numpy as np
import pytest

from tessscope.v2_1.optics.model import (
    phase_coefficients_b7,
    psf_sensor_b7,
    psf_support_energy_fraction_b7,
    unconstrained_from_coefficients_b7,
    zernike_basis_b7,
)


def test_b7_sampled_basis_is_nearly_orthonormal_on_unit_disk() -> None:
    axis = np.linspace(-1.0, 1.0, 401, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    grid = np.stack([yy, xx], axis=-1)
    inside = xx * xx + yy * yy <= 1.0
    basis = np.asarray(zernike_basis_b7(jnp.asarray(grid), 1.0))[:, inside]
    gram = basis @ basis.T / basis.shape[1]
    assert np.diag(gram) == pytest.approx(np.ones(7), abs=0.012)
    assert np.max(np.abs(gram - np.diag(np.diag(gram)))) < 0.012
    assert float(np.mean(basis[6])) == pytest.approx(0.0, abs=0.006)


def test_b7_open_ball_round_trip_and_shape_guard() -> None:
    coefficients = np.asarray([0.2, -0.1, 0.3, 0.0, 0.2, -0.2, 0.8])
    parameters = unconstrained_from_coefficients_b7(coefficients)
    recovered = np.asarray(phase_coefficients_b7(jnp.asarray(parameters)))
    assert recovered == pytest.approx(coefficients, abs=1e-6)
    assert np.linalg.norm(recovered) < 2.5
    with pytest.raises(ValueError, match="seven"):
        phase_coefficients_b7(jnp.zeros(6))


def test_b7_psf_energy_and_support() -> None:
    parameters = unconstrained_from_coefficients_b7(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    psf = np.asarray(psf_sensor_b7(jnp.asarray(parameters), jnp.asarray(6.0)))
    support = float(psf_support_energy_fraction_b7(jnp.asarray(parameters), jnp.asarray(6.0)))
    assert psf.shape == (96, 96)
    assert psf.sum() == pytest.approx(1.0, abs=1e-6)
    assert support >= 0.995
