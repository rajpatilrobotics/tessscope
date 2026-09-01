"""Generate the frozen TessScope result and demo figures.

The script reads completed gate artifacts and performs one post-evaluation,
non-cherry-picked qualitative run on the first decontaminated test source.
It never changes the frozen test report or experiment configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import torch
from evaluate_locked_test import DEPTHS_UM, load_designs, simulate_design
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from skimage.segmentation import find_boundaries

from tessscope.data.bbbc039 import ObjectNormalization, observer_size, records_for_split
from tessscope.evaluation.metrics import resample_labels
from tessscope.evaluation.route import full_source_inputs, segment_sensor_batch
from tessscope.observer.calibration import load_frozen_transform
from tessscope.observer.instanseg import load_frozen_model
from tessscope.optics.model import phase_coefficients, psf_sensor, zernike_basis

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "runs" / "gate9" / "test-report.json"
DERIVATIVE_PATH = (
    PROJECT_ROOT / "artifacts" / "runs" / "gate5" / "served-directional-derivative.json"
)

COLORS = {
    "clear": "#5C677D",
    "image_fidelity": "#D97706",
    "exact_task": "#007C91",
    "surrogate": "#7C3AED",
    "positive": "#138A5B",
    "negative": "#B33A3A",
    "grid": "#D6DAE1",
    "ink": "#172033",
    "muted": "#64748B",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_result_audit(report: dict[str, object]) -> None:
    summaries = report["summaries"]
    comparisons = report["paired_pq_comparisons"]
    photon = report["poisson_exact_vs_clear"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.25))

    depths = np.asarray([-6, -4, -2, 2, 4, 6], dtype=float)
    display = (
        ("clear", "Clear", "o"),
        ("image_fidelity", "Image fidelity", "s"),
        ("surrogate", "Surrogate VJP", "^"),
        ("exact_task", "Exact task VJP", "D"),
    )
    for key, label, marker in display:
        by_depth = summaries[key]["off_focus_pq_by_depth"]
        values = [by_depth[f"{depth:.1f}"] for depth in depths]
        axes[0].plot(
            depths,
            values,
            marker=marker,
            markersize=5,
            linewidth=2,
            color=COLORS[key],
            label=label,
        )
    axes[0].set_title("Deterministic depth sweep")
    axes[0].set_xlabel("Defocus (µm)")
    axes[0].set_ylabel("Mean panoptic quality")
    axes[0].set_xticks(depths)
    axes[0].set_ylim(0.52, 0.65)
    axes[0].legend(fontsize=8, loc="lower center", ncol=2)

    reference_keys = ("clear", "cubic", "image_fidelity", "surrogate")
    reference_labels = ("Clear", "Cubic*", "Image fidelity", "Surrogate VJP")
    y = np.arange(len(reference_keys))
    differences = np.asarray([comparisons[key]["mean_difference"] for key in reference_keys])
    lower = np.asarray([comparisons[key]["ci_lower_95"] for key in reference_keys])
    upper = np.asarray([comparisons[key]["ci_upper_95"] for key in reference_keys])
    axes[1].errorbar(
        differences,
        y,
        xerr=np.vstack([differences - lower, upper - differences]),
        fmt="D",
        color=COLORS["exact_task"],
        ecolor=COLORS["exact_task"],
        capsize=4,
        markersize=6,
        linewidth=2,
    )
    axes[1].axvline(0, color=COLORS["ink"], linewidth=1)
    axes[1].axvline(0.02, color=COLORS["muted"], linestyle="--", linewidth=1)
    axes[1].axvline(0.05, color=COLORS["negative"], linestyle=":", linewidth=1.5)
    axes[1].set_yticks(y, reference_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-0.005, 0.058)
    axes[1].set_xlabel("Exact-task minus reference PQ")
    axes[1].set_title("Paired source bootstrap (95% CI)")
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["muted"],
                linestyle="--",
                label="+0.02 baseline gates",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["negative"],
                linestyle=":",
                label="+0.05 clear gate",
            ),
        ],
        fontsize=7.5,
        loc="lower right",
    )

    levels = ("50", "200")
    values = np.asarray([photon[level]["mean_difference"] for level in levels])
    lowers = np.asarray([photon[level]["ci_lower_95"] for level in levels])
    uppers = np.asarray([photon[level]["ci_upper_95"] for level in levels])
    bar_colors = [COLORS["negative"] if value < 0 else COLORS["positive"] for value in values]
    axes[2].bar(
        np.arange(2),
        values,
        yerr=np.vstack([values - lowers, uppers - values]),
        color=bar_colors,
        alpha=0.88,
        capsize=5,
        width=0.55,
    )
    axes[2].axhline(0, color=COLORS["ink"], linewidth=1)
    axes[2].set_xticks(np.arange(2), ["50", "200"])
    axes[2].set_xlabel("Expected photons")
    axes[2].set_ylabel("Exact-task minus clear PQ")
    axes[2].set_title("Poisson endpoint robustness")
    axes[2].set_ylim(-0.0085, 0.0045)
    for index, value in enumerate(values):
        axes[2].text(index, value - 0.00065, f"{value:+.003f}", ha="center", va="top")

    fig.suptitle(
        "TessScope locked test: deterministic gain is reliable; headline gate is not met",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        -0.015,
        "41 held-out sources · 4,715 observer images · "
        "*validation selected cubic strength was 0 rad, so cubic equals clear",
        ha="center",
        color=COLORS["muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    _save(fig, "pq-results.png")


def generate_derivative_gate(derivative: dict[str, object]) -> None:
    epsilons = derivative["epsilons"]
    x = np.asarray([1e-1, 1e-2, 1e-3, 1e-4])
    relative = np.asarray([epsilons[f"{value:.0e}"]["median_relative_error"] for value in x])
    cosine = np.asarray([epsilons[f"{value:.0e}"]["cosine_agreement"] for value in x])
    stable = np.asarray([epsilons[f"{value:.0e}"]["in_stable_window"] for value in x])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    axes[0].loglog(x, relative, color=COLORS["exact_task"], marker="o", linewidth=2)
    axes[0].scatter(
        x[stable],
        relative[stable],
        s=70,
        facecolors="none",
        edgecolors=COLORS["positive"],
        linewidths=2,
    )
    axes[0].axhline(0.01, color=COLORS["negative"], linestyle="--", linewidth=1.5)
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Central-difference ε")
    axes[0].set_ylabel("Median relative error")
    axes[0].set_title("Directional derivative error")
    axes[0].text(7e-2, 0.012, "required < 0.01", color=COLORS["negative"], fontsize=8)

    axes[1].semilogx(x, cosine, color=COLORS["surrogate"], marker="D", linewidth=2)
    axes[1].scatter(
        x[stable],
        cosine[stable],
        s=70,
        facecolors="none",
        edgecolors=COLORS["positive"],
        linewidths=2,
    )
    axes[1].axhline(0.99, color=COLORS["negative"], linestyle="--", linewidth=1.5)
    axes[1].invert_xaxis()
    axes[1].set_ylim(0.25, 1.02)
    axes[1].set_xlabel("Central-difference ε")
    axes[1].set_ylabel("Cosine agreement")
    axes[1].set_title("Directional alignment")
    axes[1].text(7e-2, 0.965, "required > 0.99", color=COLORS["negative"], fontsize=8)

    fig.suptitle("Served JAX → PyTorch gradient gate passed", fontsize=14, fontweight="semibold")
    fig.text(
        0.5,
        -0.01,
        "Stable-window aggregate: relative error "
        f"{derivative['overall_median_relative_error']:.4f}; cosine "
        f"{derivative['overall_cosine_agreement']:.5f}",
        ha="center",
        color=COLORS["muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    _save(fig, "gradient-gate.png")


def generate_phase_and_psfs(designs: dict[str, dict[str, object]]) -> None:
    exact_parameters = jnp.asarray(designs["exact_task"]["phase_parameters"], dtype=jnp.float32)
    axis = jnp.linspace(-1.0, 1.0, 301)
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    grid = jnp.stack([yy, xx], axis=-1)
    basis = zernike_basis(grid, 1.0)
    coefficients = phase_coefficients(exact_parameters)
    phase = np.asarray(jnp.einsum("m,mhw->hw", coefficients, basis))
    disk = np.asarray(xx * xx + yy * yy <= 1.0)
    phase = np.where(disk, phase, np.nan)

    depths = jnp.asarray(DEPTHS_UM)
    clear_psfs = np.asarray(jax.vmap(lambda depth: psf_sensor(jnp.zeros(6), depth))(depths))
    exact_psfs = np.asarray(jax.vmap(lambda depth: psf_sensor(exact_parameters, depth))(depths))
    clear_log = np.log10(np.maximum(clear_psfs, 1e-7))
    exact_log = np.log10(np.maximum(exact_psfs, 1e-7))
    vmin = min(float(clear_log.min()), float(exact_log.min()))
    vmax = max(float(clear_log.max()), float(exact_log.max()))

    fig = plt.figure(figsize=(13.6, 4.8))
    grid_spec = fig.add_gridspec(2, 8, width_ratios=[1.8, 1, 1, 1, 1, 1, 1, 1])
    phase_ax = fig.add_subplot(grid_spec[:, 0])
    phase_image = phase_ax.imshow(phase, cmap="twilight", origin="lower", vmin=-2.1, vmax=2.1)
    phase_ax.set_title("Learned pupil phase")
    phase_ax.set_xlabel("Normalized pupil x")
    phase_ax.set_ylabel("Normalized pupil y")
    phase_ax.set_xticks([0, 150, 300], ["−1", "0", "1"])
    phase_ax.set_yticks([0, 150, 300], ["−1", "0", "1"])
    phase_ax.grid(False)
    colorbar = fig.colorbar(phase_image, ax=phase_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Phase (rad)")

    for column, depth in enumerate(np.asarray(DEPTHS_UM), start=1):
        for row, (label, values) in enumerate((("Clear", clear_log), ("Exact-task", exact_log))):
            axis_psf = fig.add_subplot(grid_spec[row, column])
            axis_psf.imshow(values[column - 1], cmap="magma", vmin=vmin, vmax=vmax, origin="lower")
            axis_psf.set_xticks([])
            axis_psf.set_yticks([])
            axis_psf.grid(False)
            if row == 0:
                axis_psf.set_title(f"{depth:+.0f} µm", fontsize=9)
            if column == 1:
                axis_psf.set_ylabel(label, fontsize=10)

    fig.suptitle(
        "Learned 1.033-rad RMS pupil and depth-dependent sensor PSFs",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.63,
        0.02,
        "PSF intensity shown on a shared log₁₀ scale; each 96×96 PSF is unit energy",
        ha="center",
        color=COLORS["muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    _save(fig, "phase-psf.png")


def _overlay(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    base = np.clip(image, 0.0, 1.0)
    rgb = np.repeat(base[..., None], 3, axis=-1)
    target_edge = find_boundaries(target, mode="inner")
    prediction_edge = find_boundaries(prediction, mode="inner")
    rgb[target_edge] = np.asarray([1.0, 0.78, 0.05])
    rgb[prediction_edge] = np.asarray([0.0, 0.86, 0.95])
    return rgb


def generate_first_source_demo(designs: dict[str, dict[str, object]]) -> str:
    record = records_for_split("test")[0]
    normalization = ObjectNormalization(125.0, 1642.0)
    object_image, target = full_source_inputs(record, normalization)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = load_frozen_transform()
    depth_indices = [0, 3, 6]

    sensors: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for key in ("clear", "exact_task"):
        complete_sensor = simulate_design(designs[key], object_image)
        sensors[key] = complete_sensor[depth_indices]
        observer_labels = segment_sensor_batch(
            model,
            sensors[key],
            observer_size(target.shape),
            device=device,
            maximum_batch_size=3,
            transform=transform,
        )
        labels[key] = resample_labels(observer_labels, target.shape)

    fig, axes = plt.subplots(4, 3, figsize=(12.2, 10.7))
    display_depths = (-6, 0, 6)
    for column, depth in enumerate(display_depths):
        axes[0, column].set_title(f"Defocus {depth:+d} µm", fontsize=11)
        for row, key in ((0, "clear"), (2, "exact_task")):
            display = np.clip((sensors[key][column] - transform.offset) / transform.scale, 0, 1)
            axes[row, column].imshow(display, cmap="gray", vmin=0, vmax=1)
            axes[row + 1, column].imshow(_overlay(display, target, labels[key][column]))
        for row in range(4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].grid(False)

    row_labels = ("Clear sensor", "Clear instances", "Exact-task sensor", "Exact-task instances")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(
        f"First frozen test source ({record.image_id}): qualitative depth sweep",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.02,
        "Yellow = target boundary · cyan = frozen InstanSeg prediction · "
        "fixed global display transform",
        ha="center",
        color=COLORS["muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, "first-source-depth-sweep.png")
    return record.image_id


def generate_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.3, 1.35, 2.1, 1.35, "Six pupil\nparameters", "#E8F3F5"),
        (3.0, 1.35, 2.25, 1.35, "JAX + Chromatix\noptics Tesseract", "#DCEFF2"),
        (5.85, 1.35, 2.0, 1.35, "Sensor image\nstack", "#F4ECDD"),
        (8.45, 1.35, 2.35, 1.35, "PyTorch + InstanSeg\nobserver Tesseract", "#EEE8F8"),
        (11.25, 1.35, 0.75, 1.35, "Task\nloss", "#E6F3EC"),
    ]
    for x, y, width, height, label, fill in boxes:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=COLORS["ink"],
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)

    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        start = (left[0] + left[2] + 0.08, 2.02)
        end = (right[0] - 0.08, 2.02)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                color=COLORS["ink"],
                linewidth=1.4,
            )
        )
    ax.text(
        6.1,
        2.95,
        "forward: physically formed images → frozen task score",
        ha="center",
        fontsize=10,
        color=COLORS["muted"],
    )
    ax.add_patch(
        FancyArrowPatch(
            (11.58, 1.1),
            (1.2, 1.1),
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.16",
            mutation_scale=14,
            color=COLORS["exact_task"],
            linewidth=2.2,
        )
    )
    ax.text(
        6.1,
        0.82,
        "backward: exact cotangent crosses PyTorch → HTTP/Tesseract boundary → JAX VJP",
        ha="center",
        color=COLORS["exact_task"],
        fontsize=10,
    )
    ax.set_title(
        "TessScope differentiable co-design path",
        fontsize=14,
        fontweight="semibold",
        pad=12,
    )
    _save(fig, "architecture.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    report = json.loads(REPORT_PATH.read_text())
    derivative = json.loads(DERIVATIVE_PATH.read_text())
    designs = load_designs()
    generate_result_audit(report)
    generate_derivative_gate(derivative)
    generate_phase_and_psfs(designs)
    first_source = generate_first_source_demo(designs)
    generate_architecture()
    print(
        json.dumps(
            {
                "status": "generated",
                "first_test_source": first_source,
                "outputs": sorted(path.name for path in OUTPUT_DIR.glob("*.png")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
