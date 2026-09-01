"""Differentiable non-task objectives used by matched optical controls."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> jax.Array:
    coordinates = jnp.arange(size, dtype=jnp.float32) - (size - 1) / 2
    kernel_1d = jnp.exp(-0.5 * jnp.square(coordinates / sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return jnp.outer(kernel_1d, kernel_1d)


def _local_mean(images: jax.Array) -> jax.Array:
    flattened = images.reshape(-1, 1, *images.shape[-2:])
    kernel = _gaussian_kernel()[None, None]
    padding = _gaussian_kernel().shape[-1] // 2
    padded = jnp.pad(
        flattened,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="reflect",
    )
    convolved = jax.lax.conv_general_dilated(
        padded,
        kernel,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return convolved.reshape(images.shape)


def structural_similarity(candidate: jax.Array, reference: jax.Array) -> jax.Array:
    """Return mean 11×11 Gaussian-window SSIM for values on the [0,1] scale."""
    candidate = jnp.asarray(candidate, dtype=jnp.float32)
    reference = jnp.asarray(reference, dtype=jnp.float32)
    if candidate.shape != reference.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    mean_candidate = _local_mean(candidate)
    mean_reference = _local_mean(reference)
    variance_candidate = _local_mean(candidate * candidate) - mean_candidate**2
    variance_reference = _local_mean(reference * reference) - mean_reference**2
    covariance = _local_mean(candidate * reference) - mean_candidate * mean_reference
    c1 = jnp.asarray(0.01**2, dtype=jnp.float32)
    c2 = jnp.asarray(0.03**2, dtype=jnp.float32)
    numerator = (2 * mean_candidate * mean_reference + c1) * (2 * covariance + c2)
    denominator = (
        mean_candidate**2 + mean_reference**2 + c1
    ) * (variance_candidate + variance_reference + c2)
    return jnp.mean(numerator / jnp.maximum(denominator, 1e-12))


def image_fidelity_loss(
    sensor: jax.Array,
    clear_focus_reference: jax.Array,
    phase_coefficients: jax.Array,
    phase_l2_weight: float = 1e-4,
) -> jax.Array:
    """Approved equally weighted NRMSE and one-minus-SSIM control loss."""
    sensor = jnp.asarray(sensor, dtype=jnp.float32)
    reference = jnp.asarray(clear_focus_reference, dtype=jnp.float32)
    if sensor.ndim != 4 or reference.shape != (sensor.shape[0], 1, *sensor.shape[-2:]):
        raise ValueError("Reference must have shape B,1,H,W for a B,D,H,W sensor")
    repeated_reference = jnp.broadcast_to(reference, sensor.shape)
    error_rms = jnp.sqrt(jnp.mean(jnp.square(sensor - repeated_reference), axis=(-2, -1)))
    reference_rms = jnp.sqrt(
        jnp.mean(jnp.square(repeated_reference), axis=(-2, -1))
    )
    nrmse = jnp.mean(error_rms / jnp.maximum(reference_rms, 1e-6))
    ssim_loss = 1.0 - structural_similarity(sensor, repeated_reference)
    regularization = phase_l2_weight * jnp.sum(jnp.square(phase_coefficients))
    return 0.5 * nrmse + 0.5 * ssim_loss + regularization
