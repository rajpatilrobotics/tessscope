import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tessscope.optics.model import (
    OpticsConfig,
    cubic_phase_basis,
    phase_coefficients,
    psf_sensor,
    psf_sensor_cubic,
    simulate_sensor,
    zernike_basis,
)


def test_phase_parameterization_respects_bound() -> None:
    coefficients = phase_coefficients(jnp.arange(1, 7, dtype=jnp.float32))
    assert float(jnp.linalg.norm(coefficients)) < 2.5


def test_zero_strength_cubic_matches_clear_psf() -> None:
    clear = psf_sensor(jnp.zeros((6,), dtype=jnp.float32), jnp.asarray(2.0))
    cubic = psf_sensor_cubic(jnp.asarray(0.0), jnp.asarray(2.0))
    np.testing.assert_allclose(np.asarray(cubic), np.asarray(clear), atol=2e-7)


def test_cubic_basis_is_approximately_rms_one_on_sampled_pupil() -> None:
    coordinates = jnp.linspace(-1.0, 1.0, 501)
    yy, xx = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    grid = jnp.stack([yy, xx], axis=-1)
    basis = np.asarray(cubic_phase_basis(grid, 1.0))
    inside = np.asarray(xx * xx + yy * yy <= 1.0)
    assert float(np.sqrt(np.mean(np.square(basis[inside])))) == pytest.approx(
        1.0, abs=0.01
    )


def test_zernike_modes_are_nearly_orthonormal_on_dense_disk() -> None:
    axis = jnp.linspace(-1, 1, 401)
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    grid = jnp.stack([yy, xx], axis=-1)
    basis = zernike_basis(grid, pupil_radius=1.0)
    inside = (xx * xx + yy * yy) <= 1
    values = basis[:, inside]
    gram = values @ values.T / values.shape[1]
    assert np.allclose(np.asarray(gram), np.eye(6), atol=0.015)


def test_psf_is_unit_energy_and_differentiable() -> None:
    parameters = jnp.zeros((6,), dtype=jnp.float32)
    psf = psf_sensor(parameters, jnp.asarray(2.0))
    gradient = jax.grad(lambda value: psf_sensor(value, jnp.asarray(2.0))[48, 48])(
        parameters
    )
    assert psf.shape == (96, 96)
    assert np.isclose(float(psf.sum()), 1.0, atol=1e-6)
    assert np.isfinite(np.asarray(gradient)).all()
    assert float(jnp.linalg.norm(gradient)) > 0


def test_sensor_batch_shape_and_finite_vjp() -> None:
    config = OpticsConfig(support_px=32, oversampling=2)
    parameters = jnp.zeros((6,), dtype=jnp.float32)
    objects = jnp.ones((2, 32, 32), dtype=jnp.float32)
    depths = jnp.asarray([-2.0, 0.0, 2.0], dtype=jnp.float32)
    output, pullback = jax.vjp(
        lambda value: simulate_sensor(value, objects, depths, config=config), parameters
    )
    (gradient,) = pullback(jnp.ones_like(output))
    assert output.shape == (2, 3, 32, 32)
    assert np.isfinite(np.asarray(output)).all()
    assert np.isfinite(np.asarray(gradient)).all()
