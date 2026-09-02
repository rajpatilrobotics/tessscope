"""Shared scientific-figure helpers for the evidence-first demo."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

from tessscope.evaluation.metrics import resample_labels  # noqa: E402

COLORS = {
    "exact": "#0072B2",
    "stopped": "#D55E00",
    "piecewise": "#009E73",
    "clear": "#7A7A7A",
    "reference": "#0072B2",
    "prediction": "#D55E00",
    "overlap": "#CC79A7",
    "ink": "#171A1F",
    "muted": "#5F6670",
    "grid": "#D9DDE3",
    "soft": "#F2F4F7",
}

LABELS = {
    "clear": "Clear",
    "exact": "Exact gradient",
    "stopped": "Stopped gradient",
    "piecewise": "Piecewise-028",
}

MARKERS = {"exact": "o", "stopped": "s", "piecewise": "^", "clear": "D"}


def apply_style() -> None:
    """Apply one restrained, deterministic publication style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "svg.hashsalt": "tessscope-evidence-first-demo-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def display_sensor(sensor: np.ndarray, scale: float) -> np.ndarray:
    """Apply the frozen global observer transform for display."""
    if scale <= 0:
        raise ValueError("Global display scale must be positive")
    return np.clip(np.asarray(sensor, dtype=np.float32) / scale, 0.0, 1.0)


def overlay_boundaries(
    sensor: np.ndarray,
    target_labels: np.ndarray,
    prediction_labels: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Overlay fixed-color reference/prediction boundaries on the sensor grid."""
    base = display_sensor(sensor, scale)
    rgb = np.repeat(base[..., None], 3, axis=-1)
    output_shape = tuple(sensor.shape[-2:])
    target = resample_labels(target_labels, output_shape)
    prediction = resample_labels(prediction_labels, output_shape)
    target_edge = find_boundaries(target, mode="inner")
    prediction_edge = find_boundaries(prediction, mode="inner")
    both = target_edge & prediction_edge
    reference_rgb = np.asarray([0.0, 114.0, 178.0]) / 255.0
    prediction_rgb = np.asarray([213.0, 94.0, 0.0]) / 255.0
    overlap_rgb = np.asarray([204.0, 121.0, 167.0]) / 255.0
    rgb[target_edge] = reference_rgb
    rgb[prediction_edge] = prediction_rgb
    rgb[both] = overlap_rgb
    return rgb


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed_key: str,
    replicates: int = 2000,
) -> tuple[float, float, float]:
    """Return deterministic percentile uncertainty over wells."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be a finite one-dimensional sample")
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little")
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(array), size=(replicates, len(array)))
    distribution = array[sampled].mean(axis=1)
    return (
        float(array.mean()),
        float(np.percentile(distribution, 2.5)),
        float(np.percentile(distribution, 97.5)),
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    """Place a consistent panel label just outside the upper-left axes corner."""
    axis.text(
        -0.12,
        1.16,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def format_depth(depth: float) -> str:
    """Format one signed depth for titles and filenames."""
    return f"{depth:+.0f} µm" if depth != 0 else "0 µm"
