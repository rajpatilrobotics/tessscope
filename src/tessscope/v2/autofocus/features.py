"""Content-normalized Fourier radial/angular features with an analytic VJP."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class SpectralFeatureConfig:
    """Fixed, design-independent spectral descriptor settings."""

    radial_bins: int = 10
    angular_bins: int = 12
    log_epsilon: float = 1e-6
    content_epsilon: float = 1e-8
    global_offset: float = 0.0
    global_scale: float = 1.0

    def validate(self) -> None:
        if self.radial_bins <= 0 or self.angular_bins <= 0:
            raise ValueError("Spectral bin counts must be positive")
        if self.log_epsilon <= 0 or self.content_epsilon <= 0:
            raise ValueError("Spectral stabilization constants must be positive")
        if self.global_scale <= 0:
            raise ValueError("Global sensor scale must be positive")


@dataclass(frozen=True)
class SpectralFeatureTape:
    """Forward values required by the hand-derived feature VJP."""

    centered: np.ndarray
    content_rms: np.ndarray
    spectrum: np.ndarray
    power: np.ndarray
    window: np.ndarray
    bin_weights: np.ndarray
    config: SpectralFeatureConfig


DEFAULT_FEATURE_CONFIG = SpectralFeatureConfig()


def _as_image_batch(images: np.ndarray) -> np.ndarray:
    value = np.asarray(images, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError(f"Expected image batch N,H,W, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Autofocus images must be finite")
    return value


@lru_cache(maxsize=16)
def fixed_window(height: int, width: int) -> np.ndarray:
    """Return a unit-RMS separable Hann window fixed by image shape."""
    if min(height, width) < 4:
        raise ValueError("Spectral autofocus requires images at least 4×4")
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float64)
    rms = float(np.sqrt(np.mean(window * window)))
    return window / rms


@lru_cache(maxsize=32)
def radial_angular_bin_weights(
    height: int,
    width: int,
    radial_bins: int,
    angular_bins: int,
) -> np.ndarray:
    """Return fixed mean-aggregation weights on the inscribed Nyquist disk."""
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    normalized_radius = radius / 0.5
    angle = np.mod(np.arctan2(fy, fx), np.pi)
    radial_index = np.minimum(
        np.floor(normalized_radius * radial_bins).astype(np.int64), radial_bins - 1
    )
    angular_index = np.minimum(
        np.floor(angle / np.pi * angular_bins).astype(np.int64), angular_bins - 1
    )
    valid = (radius > 0) & (normalized_radius <= 1.0)
    combined = radial_index * angular_bins + angular_index
    feature_count = radial_bins * angular_bins
    weights = np.zeros((feature_count, height * width), dtype=np.float64)
    flattened_bins = combined.reshape(-1)
    flattened_valid = valid.reshape(-1)
    for feature_index in range(feature_count):
        selected = flattened_valid & (flattened_bins == feature_index)
        count = int(np.count_nonzero(selected))
        if count == 0:
            raise ValueError(
                f"Empty spectral bin {feature_index} for shape {(height, width)} and "
                f"grid {(radial_bins, angular_bins)}"
            )
        weights[feature_index, selected] = 1.0 / count
    return weights


def spectral_features(
    images: np.ndarray,
    config: SpectralFeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> tuple[np.ndarray, SpectralFeatureTape]:
    """Compute stabilized features and retain an explicit analytic-VJP tape."""
    config.validate()
    value = _as_image_batch(images)
    height, width = value.shape[-2:]
    normalized = (value - config.global_offset) / config.global_scale
    centered = normalized - normalized.mean(axis=(-2, -1), keepdims=True)
    content_rms = np.sqrt(
        np.mean(centered * centered, axis=(-2, -1), keepdims=True)
        + config.content_epsilon
    )
    content_normalized = centered / content_rms
    window = fixed_window(height, width)
    spectrum = np.fft.fft2(content_normalized * window, norm="ortho")
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    log_power = np.log(power + config.log_epsilon)
    weights = radial_angular_bin_weights(
        height,
        width,
        config.radial_bins,
        config.angular_bins,
    )
    features = log_power.reshape(len(value), -1) @ weights.T
    return features, SpectralFeatureTape(
        centered=centered,
        content_rms=content_rms,
        spectrum=spectrum,
        power=power,
        window=window,
        bin_weights=weights,
        config=config,
    )


def spectral_features_vjp(
    tape: SpectralFeatureTape,
    feature_cotangent: np.ndarray,
) -> np.ndarray:
    """Apply the exact real-input adjoint of normalization/FFT/log/binning."""
    cotangent = np.asarray(feature_cotangent, dtype=np.float64)
    batch, height, width = tape.centered.shape
    expected = (batch, tape.bin_weights.shape[0])
    if cotangent.shape != expected:
        raise ValueError(f"Expected feature cotangent {expected}, got {cotangent.shape}")
    log_power_cotangent = (cotangent @ tape.bin_weights).reshape(batch, height, width)
    power_cotangent = log_power_cotangent / (
        tape.power + tape.config.log_epsilon
    )
    windowed_cotangent = 2.0 * np.fft.ifft2(
        power_cotangent * tape.spectrum,
        norm="ortho",
    ).real
    normalized_cotangent = windowed_cotangent * tape.window
    coupling = np.mean(
        normalized_cotangent * tape.centered,
        axis=(-2, -1),
        keepdims=True,
    )
    centered_cotangent = (
        normalized_cotangent / tape.content_rms
        - tape.centered * coupling / (tape.content_rms**3)
    )
    normalized_input_cotangent = centered_cotangent - centered_cotangent.mean(
        axis=(-2, -1), keepdims=True
    )
    return normalized_input_cotangent / tape.config.global_scale
