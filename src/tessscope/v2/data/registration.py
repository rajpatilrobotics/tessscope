"""Intensity-preserving z-stack registration for BBBC006 fluorescence fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation


@dataclass(frozen=True)
class RegistrationConfig:
    """Frozen numerical choices for translation-only z-stack registration."""

    downsample: int = 4
    upsample_factor: int = 20
    maximum_shift_px: float = 20.0


@dataclass(frozen=True)
class RegistrationResult:
    """Translation and before/after alignment diagnostics for one plane."""

    shift_yx_px: tuple[float, float]
    residual_shift_yx_px: tuple[float, float]
    correlation_before: float
    correlation_after: float


DEFAULT_REGISTRATION_CONFIG = RegistrationConfig()


def _registration_view(image: np.ndarray, downsample: int) -> np.ndarray:
    """Create a robust high-pass view used only to estimate translation."""
    value = np.asarray(image, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"Registration expects a 2D image, received {value.shape}")
    if downsample <= 0:
        raise ValueError("Registration downsample must be positive")
    value = np.log1p(np.maximum(value - np.percentile(value, 1.0), 0.0))
    value = value[::downsample, ::downsample]
    value -= ndimage.gaussian_filter(value, sigma=2.0)
    value -= np.mean(value)
    scale = float(np.sqrt(np.mean(np.square(value))))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Cannot register a constant or non-finite image")
    window = np.outer(np.hanning(value.shape[0]), np.hanning(value.shape[1]))
    return value * window / scale


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return zero-mean normalized correlation, with a stable constant-image policy."""
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 1.0


def apply_translation(image: np.ndarray, shift_yx_px: tuple[float, float]) -> np.ndarray:
    """Apply a subpixel translation without changing the input's global intensity scale."""
    return ndimage.shift(
        np.asarray(image, dtype=np.float64),
        shift=shift_yx_px,
        order=1,
        mode="reflect",
        prefilter=False,
    )


def register_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    config: RegistrationConfig = DEFAULT_REGISTRATION_CONFIG,
) -> RegistrationResult:
    """Estimate and audit the translation that aligns ``moving`` to ``reference``."""
    if np.shape(reference) != np.shape(moving):
        raise ValueError("Registration images must share a shape")
    reference_view = _registration_view(reference, config.downsample)
    moving_view = _registration_view(moving, config.downsample)
    shift_small, _, _ = phase_cross_correlation(
        reference_view,
        moving_view,
        upsample_factor=config.upsample_factor,
        normalization=None,
    )
    shift = np.asarray(shift_small, dtype=np.float64) * config.downsample
    if np.any(np.abs(shift) > config.maximum_shift_px):
        raise ValueError(
            f"Estimated registration shift {shift.tolist()} exceeds "
            f"{config.maximum_shift_px} px"
        )
    registered_view = ndimage.shift(
        moving_view,
        shift=shift / config.downsample,
        order=1,
        mode="reflect",
        prefilter=False,
    )
    residual_small, _, _ = phase_cross_correlation(
        reference_view,
        registered_view,
        upsample_factor=config.upsample_factor,
        normalization=None,
    )
    residual = np.asarray(residual_small, dtype=np.float64) * config.downsample
    return RegistrationResult(
        shift_yx_px=(float(shift[0]), float(shift[1])),
        residual_shift_yx_px=(float(residual[0]), float(residual[1])),
        correlation_before=normalized_correlation(reference_view, moving_view),
        correlation_after=normalized_correlation(reference_view, registered_view),
    )
