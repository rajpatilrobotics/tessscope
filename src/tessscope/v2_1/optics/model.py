"""Seven-mode B7 pupil with Noll 11 primary spherical aberration.

The original v2 B6 implementation remains unchanged. B7 keeps Noll modes 5–10
and appends RMS-normalized primary spherical, Noll 11, under the same open
2.5-radian coefficient ball.
"""

from __future__ import annotations

from functools import partial

import chromatix.functional as cx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import fftconvolve

from tessscope.optics.model import DEFAULT_OPTICS_CONFIG, OpticsConfig

B7_NOLL_INDICES = (5, 6, 7, 8, 9, 10, 11)


def phase_coefficients_b7(v: jax.Array, bound: float = 2.5) -> jax.Array:
    """Map seven unconstrained parameters into the open RMS coefficient ball."""
    value = jnp.asarray(v, dtype=jnp.float32)
    if value.shape != (7,):
        raise ValueError(f"Expected seven B7 phase parameters, got {value.shape}")
    return bound * value / jnp.sqrt(1.0 + jnp.sum(value * value))


def unconstrained_from_coefficients_b7(
    coefficients: np.ndarray,
    bound: float = 2.5,
) -> np.ndarray:
    """Invert the B7 open-ball map for an interior coefficient vector."""
    value = np.asarray(coefficients, dtype=np.float64)
    if value.shape != (7,):
        raise ValueError(f"Expected seven B7 phase coefficients, got {value.shape}")
    norm = float(np.linalg.norm(value))
    if not 0 <= norm < bound:
        raise ValueError(f"Coefficient norm must be inside the {bound}-radian RMS ball")
    return (value / np.sqrt(bound * bound - norm * norm)).astype(np.float32)


def zernike_basis_b7(grid_yx: jax.Array, pupil_radius: float) -> jax.Array:
    """Return continuous unit-disk RMS-one Noll modes 5–11."""
    y = grid_yx[..., 0] / pupil_radius
    x = grid_yx[..., 1] / pupil_radius
    radius_squared = x * x + y * y
    inside = radius_squared <= 1.0
    sqrt5 = jnp.sqrt(jnp.asarray(5.0, dtype=jnp.float32))
    sqrt6 = jnp.sqrt(jnp.asarray(6.0, dtype=jnp.float32))
    sqrt8 = jnp.sqrt(jnp.asarray(8.0, dtype=jnp.float32))
    modes = jnp.stack(
        [
            sqrt6 * (2.0 * x * y),
            sqrt6 * (x * x - y * y),
            sqrt8 * ((3.0 * radius_squared - 2.0) * y),
            sqrt8 * ((3.0 * radius_squared - 2.0) * x),
            sqrt8 * (3.0 * x * x * y - y * y * y),
            sqrt8 * (x * x * x - 3.0 * x * y * y),
            sqrt5 * (6.0 * radius_squared * radius_squared - 6.0 * radius_squared + 1.0),
        ],
        axis=0,
    )
    return jnp.where(inside[None], modes, 0.0)


def _pupil_spacing(config: OpticsConfig, simulation_size: int) -> float:
    magnification = config.tube_focal_length_um / config.objective_focal_length_um
    oversampled_camera_pitch = config.biological_sampling_um / config.oversampling * magnification
    return (
        config.tube_focal_length_um
        * config.wavelength_um
        / (config.refractive_index * simulation_size * oversampled_camera_pitch)
    )


def psf_oversampled_b7(
    v: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
    simulation_size: int | None = None,
) -> jax.Array:
    """Generate a unit-energy oversampled B7 PSF using Chromatix."""
    if simulation_size is None:
        simulation_size = config.support_px * config.oversampling
    field = cx.objective_point_source(
        shape=(simulation_size, simulation_size),
        dx=_pupil_spacing(config, simulation_size),
        spectrum=config.wavelength_um,
        z=depth_um,
        f=config.objective_focal_length_um,
        n=config.refractive_index,
        NA=config.numerical_aperture,
        power=1.0,
    )
    radius = config.objective_focal_length_um * config.numerical_aperture / config.refractive_index
    coefficients = phase_coefficients_b7(v, config.phase_bound_radians)
    phase = jnp.einsum("m,mhw->hw", coefficients, zernike_basis_b7(field.grid, radius))
    field = cx.phase_change(field, phase, spectrally_modulate=False)
    image_field = cx.ff_lens(
        field,
        f=config.tube_focal_length_um,
        n=config.refractive_index,
    )
    intensity = jnp.asarray(image_field.intensity, dtype=jnp.float32)
    return intensity / jnp.sum(intensity)


def psf_sensor_b7(
    v: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
) -> jax.Array:
    """Integrate the B7 oversampled PSF over square sensor pixels."""
    intensity = psf_oversampled_b7(v, depth_um, config)
    support = config.support_px
    factor = config.oversampling
    integrated = intensity.reshape(support, factor, support, factor).sum(axis=(1, 3))
    return integrated / jnp.sum(integrated)


def psf_support_energy_fraction_b7(
    v: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
    reference_factor: int = 2,
) -> jax.Array:
    """Measure B7 support energy against a larger same-sampling reference field."""
    base_size = config.support_px * config.oversampling
    reference_size = base_size * reference_factor
    reference = psf_oversampled_b7(
        v,
        depth_um,
        config,
        simulation_size=reference_size,
    )
    start = (reference_size - base_size) // 2
    return reference[start : start + base_size, start : start + base_size].sum()


def _same_zero_convolution(image: jax.Array, kernel: jax.Array) -> jax.Array:
    return fftconvolve(image, kernel, mode="same")


@partial(jax.jit, static_argnames=("config",))
def simulate_noisy_sensor_b7(
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
    """Simulate globally scaled B7 rates with fixed reparameterized noise."""
    objects = jnp.maximum(jnp.asarray(object_batch, dtype=jnp.float32), 0.0)
    depths = jnp.asarray(depths_um, dtype=jnp.float32) * jnp.asarray(
        depth_scale, dtype=jnp.float32
    ) + jnp.asarray(axial_offset_um, dtype=jnp.float32)
    psfs = jax.vmap(lambda depth: psf_sensor_b7(phase_parameters, depth, config))(depths)

    def image_at_depth(psf: jax.Array) -> jax.Array:
        return jax.vmap(lambda image: _same_zero_convolution(image, psf))(objects)

    rate = jax.vmap(image_at_depth)(psfs).transpose(1, 0, 2, 3) * jnp.asarray(
        exposure_gain, dtype=jnp.float32
    )
    expected_counts = jnp.asarray(expected_photons, dtype=jnp.float32) * rate
    noise = jnp.asarray(noise_standard_normal, dtype=jnp.float32)
    if noise.shape != expected_counts.shape:
        raise ValueError(
            f"Noise shape {noise.shape} differs from sensor shape {expected_counts.shape}"
        )
    noisy_counts = expected_counts + jnp.sqrt(jnp.maximum(expected_counts, 0.0) + 1e-6) * noise
    return jnp.maximum(noisy_counts, 0.0) / jnp.asarray(expected_photons, dtype=jnp.float32)
