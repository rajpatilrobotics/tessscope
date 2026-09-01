"""Six-mode phase-only pupil and incoherent image formation.

Chromatix constructs the defocused objective pupil, applies the phase mask, and
propagates it through the tube lens. JAX handles batching, convolution, and VJPs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import chromatix.functional as cx
import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve


@dataclass(frozen=True)
class OpticsConfig:
    wavelength_um: float = 0.461
    numerical_aperture: float = 0.45
    refractive_index: float = 1.0
    objective_focal_length_um: float = 20_000.0
    tube_focal_length_um: float = 200_000.0
    biological_sampling_um: float = 0.645
    oversampling: int = 4
    support_px: int = 96
    phase_bound_radians: float = 2.5


DEFAULT_OPTICS_CONFIG = OpticsConfig()


def phase_coefficients(v: jax.Array, bound: float = 2.5) -> jax.Array:
    """Map unconstrained parameters into the frozen open RMS ball."""
    v = jnp.asarray(v, dtype=jnp.float32)
    if v.shape != (6,):
        raise ValueError(f"Expected six phase parameters, got {v.shape}")
    return bound * v / jnp.sqrt(1.0 + jnp.sum(v * v))


def zernike_basis(grid_yx: jax.Array, pupil_radius: float) -> jax.Array:
    """Return orthonormal unit-disk Noll modes 5–10.

    Sign/orientation choices only rotate modes within the same frozen span. The
    normalization makes the coefficient vector norm equal pupil phase RMS in
    the continuous unit-disk convention.
    """
    y = grid_yx[..., 0] / pupil_radius
    x = grid_yx[..., 1] / pupil_radius
    radius_squared = x * x + y * y
    inside = radius_squared <= 1.0
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
        ],
        axis=0,
    )
    return jnp.where(inside[None], modes, 0.0)


def cubic_phase_basis(grid_yx: jax.Array, pupil_radius: float) -> jax.Array:
    """Return RMS-one separable cubic phase ``x³+y³`` on the unit disk."""
    y = grid_yx[..., 0] / pupil_radius
    x = grid_yx[..., 1] / pupil_radius
    inside = x * x + y * y <= 1.0
    # E_disk[x^6 + y^6] = 5/32 and E_disk[x^3 y^3] = 0.
    normalization = jnp.sqrt(jnp.asarray(5.0 / 32.0, dtype=jnp.float32))
    basis = (x * x * x + y * y * y) / normalization
    return jnp.where(inside, basis, 0.0)


def _pupil_spacing(config: OpticsConfig, simulation_size: int) -> float:
    magnification = config.tube_focal_length_um / config.objective_focal_length_um
    oversampled_camera_pitch = (
        config.biological_sampling_um / config.oversampling * magnification
    )
    return (
        config.tube_focal_length_um
        * config.wavelength_um
        / (config.refractive_index * simulation_size * oversampled_camera_pitch)
    )


def psf_oversampled(
    v: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
    simulation_size: int | None = None,
) -> jax.Array:
    """Generate a unit-energy oversampled PSF using Chromatix."""
    if simulation_size is None:
        simulation_size = config.support_px * config.oversampling
    pupil_spacing = _pupil_spacing(config, simulation_size)
    field = cx.objective_point_source(
        shape=(simulation_size, simulation_size),
        dx=pupil_spacing,
        spectrum=config.wavelength_um,
        z=depth_um,
        f=config.objective_focal_length_um,
        n=config.refractive_index,
        NA=config.numerical_aperture,
        power=1.0,
    )
    radius = (
        config.objective_focal_length_um
        * config.numerical_aperture
        / config.refractive_index
    )
    coefficients = phase_coefficients(v, config.phase_bound_radians)
    phase = jnp.einsum("m,mhw->hw", coefficients, zernike_basis(field.grid, radius))
    field = cx.phase_change(field, phase, spectrally_modulate=False)
    image_field = cx.ff_lens(
        field,
        f=config.tube_focal_length_um,
        n=config.refractive_index,
    )
    intensity = jnp.asarray(image_field.intensity, dtype=jnp.float32)
    return intensity / jnp.sum(intensity)


def psf_sensor(
    v: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
) -> jax.Array:
    """Integrate the oversampled PSF over square sensor pixels."""
    intensity = psf_oversampled(v, depth_um, config)
    support = config.support_px
    factor = config.oversampling
    integrated = intensity.reshape(support, factor, support, factor).sum(axis=(1, 3))
    return integrated / jnp.sum(integrated)


def psf_oversampled_cubic(
    rms_strength_radians: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
    simulation_size: int | None = None,
) -> jax.Array:
    """Generate a cubic-mask PSF with the requested pupil RMS phase."""
    if simulation_size is None:
        simulation_size = config.support_px * config.oversampling
    pupil_spacing = _pupil_spacing(config, simulation_size)
    field = cx.objective_point_source(
        shape=(simulation_size, simulation_size),
        dx=pupil_spacing,
        spectrum=config.wavelength_um,
        z=depth_um,
        f=config.objective_focal_length_um,
        n=config.refractive_index,
        NA=config.numerical_aperture,
        power=1.0,
    )
    radius = (
        config.objective_focal_length_um
        * config.numerical_aperture
        / config.refractive_index
    )
    phase = jnp.asarray(rms_strength_radians, dtype=jnp.float32) * cubic_phase_basis(
        field.grid, radius
    )
    field = cx.phase_change(field, phase, spectrally_modulate=False)
    image_field = cx.ff_lens(
        field,
        f=config.tube_focal_length_um,
        n=config.refractive_index,
    )
    intensity = jnp.asarray(image_field.intensity, dtype=jnp.float32)
    return intensity / jnp.sum(intensity)


def psf_sensor_cubic(
    rms_strength_radians: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
) -> jax.Array:
    """Integrate an RMS-normalized cubic PSF over square sensor pixels."""
    intensity = psf_oversampled_cubic(rms_strength_radians, depth_um, config)
    support = config.support_px
    factor = config.oversampling
    integrated = intensity.reshape(support, factor, support, factor).sum(axis=(1, 3))
    return integrated / jnp.sum(integrated)


def cubic_support_energy_fraction(
    rms_strength_radians: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
    reference_factor: int = 2,
) -> jax.Array:
    """Measure cubic-mask support energy against a larger reference field."""
    base_size = config.support_px * config.oversampling
    reference_size = base_size * reference_factor
    reference = psf_oversampled_cubic(
        rms_strength_radians,
        depth_um,
        config,
        simulation_size=reference_size,
    )
    start = (reference_size - base_size) // 2
    return reference[start : start + base_size, start : start + base_size].sum()


def psf_support_energy_fraction(
    v: jax.Array,
    depth_um: jax.Array,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
    reference_factor: int = 2,
) -> jax.Array:
    """Measure support energy against a larger same-sampling reference."""
    base_size = config.support_px * config.oversampling
    reference_size = base_size * reference_factor
    reference = psf_oversampled(v, depth_um, config, simulation_size=reference_size)
    start = (reference_size - base_size) // 2
    return reference[start : start + base_size, start : start + base_size].sum()


def _same_zero_convolution(image: jax.Array, kernel: jax.Array) -> jax.Array:
    """FFT convolution with SciPy-compatible same-size zero boundaries."""
    return fftconvolve(image, kernel, mode="same")


@partial(jax.jit, static_argnames=("config",))
def simulate_sensor(
    v: jax.Array,
    object_batch: jax.Array,
    depths_um: jax.Array,
    expected_photons: float = 100.0,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
) -> jax.Array:
    """Return deterministic normalized sensor rates with shape B,D,H,W."""
    objects = jnp.asarray(object_batch, dtype=jnp.float32)
    depths = jnp.asarray(depths_um, dtype=jnp.float32)
    psfs = jax.vmap(lambda depth: psf_sensor(v, depth, config))(depths)

    def image_at_depth(psf: jax.Array) -> jax.Array:
        return jax.vmap(lambda image: _same_zero_convolution(image, psf))(objects)

    normalized_rate = jax.vmap(image_at_depth)(psfs).transpose(1, 0, 2, 3)
    expected_counts = jnp.asarray(expected_photons, dtype=jnp.float32) * normalized_rate
    return expected_counts / jnp.asarray(expected_photons, dtype=jnp.float32)


@partial(jax.jit, static_argnames=("config",))
def simulate_cubic_sensor(
    rms_strength_radians: jax.Array,
    object_batch: jax.Array,
    depths_um: jax.Array,
    expected_photons: float = 100.0,
    config: OpticsConfig = DEFAULT_OPTICS_CONFIG,
) -> jax.Array:
    """Return deterministic cubic-mask sensor rates with shape B,D,H,W."""
    objects = jnp.asarray(object_batch, dtype=jnp.float32)
    depths = jnp.asarray(depths_um, dtype=jnp.float32)
    psfs = jax.vmap(
        lambda depth: psf_sensor_cubic(rms_strength_radians, depth, config)
    )(depths)

    def image_at_depth(psf: jax.Array) -> jax.Array:
        return jax.vmap(lambda image: _same_zero_convolution(image, psf))(objects)

    normalized_rate = jax.vmap(image_at_depth)(psfs).transpose(1, 0, 2, 3)
    expected_counts = jnp.asarray(expected_photons, dtype=jnp.float32) * normalized_rate
    return expected_counts / jnp.asarray(expected_photons, dtype=jnp.float32)
