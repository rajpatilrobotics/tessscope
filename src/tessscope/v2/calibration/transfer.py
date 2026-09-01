"""Radial OTF transfer summaries and bounded curve calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransferFit:
    """One through-origin scale fit between simulated and observed transfers."""

    amplitude: float
    correlation: float
    root_mean_squared_error: float


def radial_frequency_indices(
    height: int, width: int, radial_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin indices and the inscribed-Nyquist validity mask."""
    if min(height, width) <= 0 or radial_bins <= 0:
        raise ValueError("Frequency grid dimensions and bins must be positive")
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    normalized_radius = np.sqrt(fx * fx + fy * fy) / 0.5
    indices = np.minimum(
        np.floor(normalized_radius * radial_bins).astype(np.int64), radial_bins - 1
    )
    valid = (normalized_radius > 0) & (normalized_radius <= 1.0)
    return indices, valid


def radial_otf_log_power(
    psfs: np.ndarray,
    *,
    output_size: int = 256,
    radial_bins: int = 10,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Summarize unit-DC OTF log power for centered square PSFs."""
    value = np.asarray(psfs, dtype=np.float64)
    if value.ndim != 3 or value.shape[-2] != value.shape[-1]:
        raise ValueError(f"PSFs must have shape N,S,S, received {value.shape}")
    support = value.shape[-1]
    if support > output_size or epsilon <= 0:
        raise ValueError("OTF output size must contain the PSF and epsilon must be positive")
    padded = np.zeros((len(value), output_size, output_size), dtype=np.float64)
    start = (output_size - support) // 2
    padded[:, start : start + support, start : start + support] = value
    optical_transfer = np.fft.fft2(np.fft.ifftshift(padded, axes=(-2, -1)))
    dc = np.maximum(np.abs(optical_transfer[:, :1, :1]), epsilon)
    log_power = np.log(np.maximum(np.square(np.abs(optical_transfer / dc)), epsilon))
    indices, valid = radial_frequency_indices(output_size, output_size, radial_bins)
    result = np.empty((len(value), radial_bins), dtype=np.float64)
    for index in range(radial_bins):
        selected = valid & (indices == index)
        if not np.any(selected):
            raise ValueError(f"Empty radial OTF bin {index}")
        result[:, index] = np.mean(log_power[:, selected], axis=1)
    return result


def transfer_ratios(curves: np.ndarray, focus_index: int = 3) -> np.ndarray:
    """Subtract the focus-plane curve to cancel shared object-spectrum structure."""
    value = np.asarray(curves, dtype=np.float64)
    if value.ndim != 2 or not 0 <= focus_index < len(value):
        raise ValueError("Transfer curves must be D,F with a valid focus index")
    return value - value[focus_index]


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson correlation with explicit nonconstant input validation."""
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Correlation inputs must be equal arrays with at least two values")
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        raise ValueError("Correlation is undefined for a constant curve")
    return float(np.dot(left, right) / denominator)


def fit_transfer_amplitude(observed: np.ndarray, simulated: np.ndarray) -> TransferFit:
    """Fit one nonnegative scale without hiding a frequency/depth-dependent mismatch."""
    target = np.asarray(observed, dtype=np.float64).reshape(-1)
    predictor = np.asarray(simulated, dtype=np.float64).reshape(-1)
    if target.shape != predictor.shape or target.size < 2:
        raise ValueError("Transfer fit inputs must contain matching values")
    denominator = float(np.dot(predictor, predictor))
    if denominator <= 0:
        raise ValueError("Simulated transfer is identically zero")
    amplitude = max(0.0, float(np.dot(predictor, target) / denominator))
    fitted = amplitude * predictor
    return TransferFit(
        amplitude=amplitude,
        correlation=pearson_correlation(target, predictor),
        root_mean_squared_error=float(np.sqrt(np.mean(np.square(fitted - target)))),
    )
