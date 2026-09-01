"""V2 bounded optics, reparameterized training noise, and true Poisson evaluation."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import fftconvolve

from tessscope.optics.model import (
    DEFAULT_OPTICS_CONFIG,
    OpticsConfig,
    psf_sensor,
    psf_sensor_cubic,
)


def unconstrained_from_coefficients(
    coefficients: np.ndarray,
    bound: float = 2.5,
) -> np.ndarray:
    """Invert the open-ball parameterization for any interior coefficient vector."""
    value = np.asarray(coefficients, dtype=np.float64)
    if value.shape != (6,):
        raise ValueError(f"Expected six phase coefficients, received {value.shape}")
    norm = float(np.linalg.norm(value))
    if not 0 <= norm < bound:
        raise ValueError(f"Coefficient norm must be inside the {bound}-radian RMS ball")
    return (value / np.sqrt(bound * bound - norm * norm)).astype(np.float32)


def astigmatic_parameters(
    rms_strength_radians: float,
    *,
    orientation: int = 0,
    bound: float = 2.5,
) -> np.ndarray:
    """Return parameters for one of the two RMS-normalized astigmatism modes."""
    if orientation not in {0, 1}:
        raise ValueError("Astigmatism orientation must select mode 0 or 1")
    coefficients = np.zeros(6, dtype=np.float64)
    coefficients[orientation] = rms_strength_radians
    return unconstrained_from_coefficients(coefficients, bound)


def _same_zero_convolution(image: jax.Array, kernel: jax.Array) -> jax.Array:
    return fftconvolve(image, kernel, mode="same")


@partial(jax.jit, static_argnames=("config",))
def simulate_noisy_sensor(
    phase_parameters: jax.Array,
    object_batch: jax.Array,
    depths_um: jax.Array,
    noise_standard_normal: jax.Array,
    *,
    expected_photons: float,
    exposure_gain: float,
    depth_scale: float,
    axial_offset_um: float = 0.0,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
) -> jax.Array:
    """Simulate globally scaled rates with a fixed Gaussian Poisson approximation."""
    objects = jnp.maximum(jnp.asarray(object_batch, dtype=jnp.float32), 0.0)
    depths = (
        jnp.asarray(depths_um, dtype=jnp.float32)
        * jnp.asarray(depth_scale, dtype=jnp.float32)
        + jnp.asarray(axial_offset_um, dtype=jnp.float32)
    )
    psfs = jax.vmap(lambda depth: psf_sensor(phase_parameters, depth, config))(depths)

    def image_at_depth(psf: jax.Array) -> jax.Array:
        return jax.vmap(lambda image: _same_zero_convolution(image, psf))(objects)

    rate = (
        jax.vmap(image_at_depth)(psfs).transpose(1, 0, 2, 3)
        * jnp.asarray(exposure_gain, dtype=jnp.float32)
    )
    expected_counts = jnp.asarray(expected_photons, dtype=jnp.float32) * rate
    noise = jnp.asarray(noise_standard_normal, dtype=jnp.float32)
    if noise.shape != expected_counts.shape:
        raise ValueError(
            f"Noise shape {noise.shape} differs from sensor shape {expected_counts.shape}"
        )
    noisy_counts = expected_counts + jnp.sqrt(jnp.maximum(expected_counts, 0.0) + 1e-6) * noise
    return jnp.maximum(noisy_counts, 0.0) / jnp.asarray(
        expected_photons, dtype=jnp.float32
    )


@partial(jax.jit, static_argnames=("config",))
def simulate_cubic_rate(
    rms_strength_radians: jax.Array,
    object_batch: jax.Array,
    depths_um: jax.Array,
    *,
    exposure_gain: float,
    depth_scale: float,
    axial_offset_um: float = 0.0,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
) -> jax.Array:
    """Simulate the matched cubic baseline without stochastic training noise."""
    objects = jnp.maximum(jnp.asarray(object_batch, dtype=jnp.float32), 0.0)
    depths = (
        jnp.asarray(depths_um, dtype=jnp.float32) * depth_scale + axial_offset_um
    )
    psfs = jax.vmap(
        lambda depth: psf_sensor_cubic(rms_strength_radians, depth, config)
    )(depths)

    def image_at_depth(psf: jax.Array) -> jax.Array:
        return jax.vmap(lambda image: _same_zero_convolution(image, psf))(objects)

    return (
        jax.vmap(image_at_depth)(psfs).transpose(1, 0, 2, 3) * exposure_gain
    )


def true_poisson_sensor(
    deterministic_rate: np.ndarray,
    expected_photons: float,
    *,
    seed: int,
) -> np.ndarray:
    """Sample true Poisson counts and return them on the common rate scale."""
    rate = np.asarray(deterministic_rate, dtype=np.float64)
    if expected_photons <= 0 or np.any(rate < 0) or not np.isfinite(rate).all():
        raise ValueError("Poisson rates must be finite/nonnegative and photons positive")
    generator = np.random.default_rng(seed)
    return (
        generator.poisson(expected_photons * rate).astype(np.float32)
        / expected_photons
    )
