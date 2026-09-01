"""V2 optics parameterization and photon-noise checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tessscope.optics.model import phase_coefficients
from tessscope.v2.optics.model import (
    astigmatic_parameters,
    true_poisson_sensor,
    unconstrained_from_coefficients,
)


def test_unconstrained_inverse_recovers_phase_coefficients() -> None:
    coefficients = np.asarray([0.5, -0.25, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    parameters = unconstrained_from_coefficients(coefficients)

    recovered = np.asarray(phase_coefficients(jnp.asarray(parameters)))

    assert np.allclose(recovered, coefficients, atol=2e-7)
    assert np.linalg.norm(recovered) < 2.5


def test_astigmatic_parameters_select_one_normalized_mode() -> None:
    parameters = astigmatic_parameters(1.25, orientation=1)
    coefficients = np.asarray(phase_coefficients(jnp.asarray(parameters)))

    assert np.allclose(coefficients, [0, 1.25, 0, 0, 0, 0], atol=2e-7)


def test_true_poisson_sensor_is_reproducible_and_positive() -> None:
    rate = np.full((2, 3, 8, 8), 0.5)

    first = true_poisson_sensor(rate, 50.0, seed=41)
    second = true_poisson_sensor(rate, 50.0, seed=41)

    assert np.array_equal(first, second)
    assert float(np.mean(first) * 50.0) > 0


def test_phase_parameterization_has_finite_gradient() -> None:
    gradient = jax.grad(lambda value: jnp.sum(phase_coefficients(value) ** 2))(
        jnp.ones(6, dtype=jnp.float32)
    )
    assert np.isfinite(np.asarray(gradient)).all()
