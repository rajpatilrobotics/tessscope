"""Matched deterministic and Poisson photon-budget conversions."""

from __future__ import annotations

import hashlib

import numpy as np


def expected_counts(normalized_rate: np.ndarray, expected_photons: float) -> np.ndarray:
    """Convert a nonnegative normalized sensor rate to expected photon counts."""
    if expected_photons <= 0:
        raise ValueError("Expected photons must be positive")
    rate = np.asarray(normalized_rate, dtype=np.float32)
    if np.any(rate < 0):
        raise ValueError("Normalized sensor rates must be nonnegative")
    return rate * np.float32(expected_photons)


def normalized_counts(counts: np.ndarray, expected_photons: float) -> np.ndarray:
    """Map photon counts back to the rate scale used by the frozen observer."""
    if expected_photons <= 0:
        raise ValueError("Expected photons must be positive")
    return np.asarray(counts, dtype=np.float32) / np.float32(expected_photons)


def keyed_poisson_seed(
    source_image_id: str,
    depth_um: float,
    expected_photons: int,
    replicate: int,
) -> int:
    """Derive a stable RNG seed without depending on dataset iteration order."""
    payload = f"{source_image_id}|{depth_um:.6f}|{expected_photons}|{replicate}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def sample_poisson_rate(
    normalized_rate: np.ndarray,
    expected_photons: int,
    *,
    source_image_id: str,
    depth_um: float,
    replicate: int,
) -> np.ndarray:
    """Draw a reproducible Poisson realization and return it on the rate scale."""
    counts = expected_counts(normalized_rate, expected_photons)
    seed = keyed_poisson_seed(
        source_image_id, depth_um, expected_photons, replicate
    )
    generator = np.random.default_rng(seed)
    sampled = generator.poisson(counts).astype(np.float32)
    return normalized_counts(sampled, expected_photons)
