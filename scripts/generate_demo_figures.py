"""Generate publication figures and depth replay from frozen validation evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from tessscope.demo.evidence import EXACT_NAME, PIECEWISE_NAME, STOPPED_NAME, sha256_path
from tessscope.demo.figures import (
    COLORS,
    LABELS,
    MARKERS,
    add_panel_label,
    apply_style,
    bootstrap_mean_interval,
    format_depth,
    overlay_boundaries,
)
from tessscope.evaluation.metrics import resample_labels

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = PROJECT_ROOT / "artifacts" / "runs" / "demo" / "validation-replay.npz"
REPLAY_METADATA_PATH = (
    PROJECT_ROOT / "artifacts" / "runs" / "demo" / "validation-replay.json"
)
V2_4_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)
PIECEWISE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "validation"
    / "piecewise-hard-frontier.json"
)
DERIVATIVE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_3"
    / "gates"
    / "closed-loop-derivative.json"
)
CLAIMS_PATH = PROJECT_ROOT / "configs" / "demo" / "claim-matrix.yaml"
VISUAL_PATH = PROJECT_ROOT / "configs" / "demo" / "visual-contract.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "demo"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
REPLAY_ROOT = OUTPUT_ROOT / "replay"
MANIFEST_PATH = OUTPUT_ROOT / "figure-manifest.json"
CAPTIONS_PATH = OUTPUT_ROOT / "CAPTIONS.md"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _save_figure(fig: plt.Figure, stem: str, description: str) -> dict[str, dict]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    result = {}
    metadata = {
        "png": {"Software": "TessScope", "Description": description},
        "svg": {"Date": None, "Creator": "TessScope"},
        "pdf": {"CreationDate": None, "ModDate": None, "Creator": "TessScope"},
    }
    for suffix in ("png", "svg", "pdf"):
        path = FIGURE_ROOT / f"{stem}.{suffix}"
        options = {"bbox_inches": "tight", "facecolor": "white", "metadata": metadata[suffix]}
        if suffix == "png":
            options["dpi"] = 300
        fig.savefig(path, **options)
        result[suffix] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
    plt.close(fig)
    return result


def _frame(metadata: dict, system: str, depth: float) -> dict:
    matches = [
        row
        for row in metadata["frames"]
        if row["system"] == system and row["depth_um"] == depth
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one traced frame for {system} at {depth}")
    return matches[0]


def _draw_overlay_panel(
    axis: plt.Axes,
    sensor: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    scale: float,
    title: str,
    subtitle: str,
) -> None:
    axis.imshow(
        overlay_boundaries(sensor, target, prediction, scale=scale),
        interpolation="nearest",
    )
    axis.set_title(title, pad=5)
    axis.text(
        0.5,
        -0.05,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)


def generate_matched_microscopy(arrays: dict[str, np.ndarray], metadata: dict) -> tuple[dict, dict]:
    depth = float(metadata["selected_display_depth_um"])
    depth_index = metadata["depths_um"].index(depth)
    scale = float(metadata["observer"]["transform"]["scale"])
    target = arrays["target_labels"]
    system_index = {name: index for index, name in enumerate(metadata["system_order"])}
    corrected_index = {
        name: index for index, name in enumerate(metadata["corrected_system_order"])
    }

    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.2), constrained_layout=False)
    for column, system in enumerate(metadata["system_order"]):
        if system == "clear":
            subtitle = "same field; qualitative clear reference"
        else:
            trace = _frame(metadata, system, depth)
            subtitle = f"PQ {trace['before_pq']:.3f}"
        _draw_overlay_panel(
            axes[0, column],
            arrays["sensor_before"][system_index[system], depth_index],
            target,
            arrays["labels_before"][system_index[system], depth_index],
            scale=scale,
            title=LABELS[system],
            subtitle=subtitle,
        )

    object_display = np.clip(arrays["object_image"], 0.0, 1.0)
    reference = resample_labels(target, object_display.shape)
    boundary = np.zeros((*object_display.shape, 3), dtype=np.float32)
    boundary[:] = object_display[..., None]
    from skimage.segmentation import find_boundaries

    boundary[find_boundaries(reference, mode="inner")] = np.asarray([0.0, 114.0, 178.0]) / 255
    axes[1, 0].imshow(boundary, interpolation="nearest")
    axes[1, 0].set_title("Automated reference", pad=5)
    axes[1, 0].text(
        0.5,
        -0.05,
        "BBBC006/CellProfiler instances",
        transform=axes[1, 0].transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color=COLORS["muted"],
    )
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1, 0].grid(False)
    for column, system in enumerate(metadata["corrected_system_order"], start=1):
        trace = _frame(metadata, system, depth)
        _draw_overlay_panel(
            axes[1, column],
            arrays["sensor_corrected"][corrected_index[system], depth_index],
            target,
            arrays["labels_corrected"][corrected_index[system], depth_index],
            scale=scale,
            title=f"{LABELS[system]} corrected",
            subtitle=f"residual {trace['residual_depth_um']:+.2f} µm · PQ {trace['after_pq']:.3f}",
        )

    axes[0, 0].text(
        -0.13,
        0.5,
        f"First frame\n{format_depth(depth)}",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=8,
        fontweight="semibold",
    )
    axes[1, 0].text(
        -0.13,
        0.5,
        "After one\nstage action",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=8,
        fontweight="semibold",
    )
    legend = [
        Line2D([0], [0], color=COLORS["reference"], lw=2, label="Reference boundary"),
        Line2D([0], [0], color=COLORS["prediction"], lw=2, label="InstanSeg boundary"),
        Line2D([0], [0], color=COLORS["overlap"], lw=2, label="Boundary overlap"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        "Representative validation field: matched first and corrected frames",
        fontsize=14,
        fontweight="semibold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.93,
        "n21_s1 · frozen crop (0, 220) · global display transform · no scale bar",
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.10, wspace=0.08, hspace=0.22)
    caption = (
        "Representative hard-density validation field n21_s1 at −2 µm, selected before "
        "rendering by the frozen median-distance rule. Every system uses the identical crop, "
        "geometry, nearest-neighbor label interpolation, and global observer transform. Blue "
        "boundaries are automated BBBC006/CellProfiler references; orange boundaries are "
        "official InstanSeg outputs. The representative frame is not the population effect."
    )
    alt = (
        "Two-row matched microscopy grid for validation field n21_s1. The top row compares "
        "clear, exact-gradient, stopped-gradient, and piecewise first frames at minus two "
        "micrometres. The bottom row shows the automated reference and corrected exact, "
        "stopped, and piecewise frames with reference and InstanSeg boundaries."
    )
    return _save_figure(fig, "matched-microscopy", alt), {"caption": caption, "alt": alt}


def generate_pupil_psf(arrays: dict[str, np.ndarray], metadata: dict) -> tuple[dict, dict]:
    phase = arrays["pupil_phase_radians"]
    mask = arrays["pupil_mask"].astype(bool)
    wrapped = np.angle(np.exp(1j * phase)).astype(np.float32)
    wrapped[:, ~mask] = np.nan
    psf = arrays["psf_sensor"]
    crop_size = 41
    crop_start = (psf.shape[-1] - crop_size) // 2
    cropped = psf[:, :, crop_start : crop_start + crop_size, crop_start : crop_start + crop_size]
    global_maximum = float(psf.max())
    log_psf = np.log10(np.maximum(cropped / global_maximum, 1e-5))

    fig, axes = plt.subplots(4, 8, figsize=(15.4, 7.8))
    phase_image = None
    psf_image = None
    for row, system in enumerate(metadata["system_order"]):
        phase_image = axes[row, 0].imshow(
            wrapped[row], cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest"
        )
        axes[row, 0].set_ylabel(LABELS[system], fontweight="semibold")
        axes[row, 0].set_title("Wrapped pupil phase" if row == 0 else "")
        for column, depth in enumerate(metadata["depths_um"], start=1):
            psf_image = axes[row, column].imshow(
                log_psf[row, column - 1],
                cmap="magma",
                vmin=-5.0,
                vmax=0.0,
                interpolation="nearest",
            )
            axes[row, column].set_title(format_depth(depth) if row == 0 else "")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
    assert phase_image is not None and psf_image is not None
    phase_bar_axis = fig.add_axes([0.085, 0.048, 0.105, 0.014])
    phase_bar = fig.colorbar(phase_image, cax=phase_bar_axis, orientation="horizontal")
    phase_bar.set_label("Wrapped phase (rad)")
    phase_bar.set_ticks([-np.pi, 0, np.pi])
    phase_bar.set_ticklabels(["−π", "0", "+π"])
    psf_bar_axis = fig.add_axes([0.925, 0.25, 0.012, 0.50])
    psf_bar = fig.colorbar(psf_image, cax=psf_bar_axis)
    psf_bar.set_label("log₁₀ intensity relative to global maximum")
    fig.suptitle(
        "Frozen phase masks and Chromatix sensor-plane PSFs across depth",
        fontsize=14,
        fontweight="semibold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "All PSFs use the same 41 × 41 central sensor-pixel crop and fixed [−5, 0] log scale.",
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.91, top=0.91, bottom=0.09, wspace=0.08, hspace=0.12)
    caption = (
        "Wrapped phase masks and unit-energy Chromatix/JAX sensor-plane PSFs for clear, "
        "exact-gradient, forward-identical stopped-gradient, and piecewise-028 systems. "
        "PSFs span the complete frozen −6 to +6 µm depth grid with one global log-intensity "
        "scale and identical central crop."
    )
    alt = (
        "Four-row optical comparison. Each row begins with a circular wrapped phase mask and "
        "continues with seven PSF images from minus six through plus six micrometres. Clear, "
        "exact, stopped, and piecewise masks create visibly different depth-dependent PSFs."
    )
    details = {
        "caption": caption,
        "alt": alt,
        "psf_display": {
            "source_shape_px": [96, 96],
            "crop_start_yx": [crop_start, crop_start],
            "crop_shape_px": [crop_size, crop_size],
            "global_maximum": global_maximum,
            "formula": "log10(max(psf/global_maximum, 1e-5))",
            "limits": [-5.0, 0.0],
        },
    }
    return _save_figure(fig, "pupil-psf-depth", alt), details


def _curve(
    rows: list[dict],
    *,
    design: str,
    depth_key: str,
    value_key: str,
    depths: list[float],
    seed_prefix: str,
) -> list[dict]:
    output = []
    for depth in depths:
        values = [
            float(row[value_key])
            for row in rows
            if row["design"] == design
            and row.get("hard_dense_patch", True)
            and float(row[depth_key]) == depth
        ]
        if len(values) != 27:
            raise ValueError(f"Expected 27 wells for {design} {depth} {value_key}")
        mean, lower, upper = bootstrap_mean_interval(
            values, seed_key=f"{seed_prefix}:{design}:{depth}:{value_key}"
        )
        output.append(
            {
                "depth_um": depth,
                "mean": mean,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "well_count": len(values),
                "bootstrap_replicates": 2000,
            }
        )
    return output


def generate_depth_curves(v2_4: dict, piecewise: dict) -> tuple[dict, dict]:
    depths = [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0]
    pq_curves = {
        "exact_before": _curve(
            v2_4["rows"],
            design=EXACT_NAME,
            depth_key="depth_um",
            value_key="panoptic_quality",
            depths=depths,
            seed_prefix="demo-pq-before-v1",
        ),
        "exact_after": _curve(
            v2_4["corrected_rows"],
            design=EXACT_NAME,
            depth_key="original_depth_um",
            value_key="after_pq",
            depths=depths,
            seed_prefix="demo-pq-after-v1",
        ),
        "stopped_after": _curve(
            v2_4["corrected_rows"],
            design=STOPPED_NAME,
            depth_key="original_depth_um",
            value_key="after_pq",
            depths=depths,
            seed_prefix="demo-pq-after-v1",
        ),
        "piecewise_after": _curve(
            piecewise["corrected_rows"],
            design=PIECEWISE_NAME,
            depth_key="original_depth_um",
            value_key="after_pq",
            depths=depths,
            seed_prefix="demo-pq-after-v1",
        ),
    }
    focus_curves = {
        "exact": _curve(
            [
                {**row, "absolute_residual_depth_um": abs(row["residual_depth_um"])}
                for row in v2_4["corrected_rows"]
            ],
            design=EXACT_NAME,
            depth_key="original_depth_um",
            value_key="absolute_residual_depth_um",
            depths=depths,
            seed_prefix="demo-focus-v1",
        ),
        "stopped": _curve(
            [
                {**row, "absolute_residual_depth_um": abs(row["residual_depth_um"])}
                for row in v2_4["corrected_rows"]
            ],
            design=STOPPED_NAME,
            depth_key="original_depth_um",
            value_key="absolute_residual_depth_um",
            depths=depths,
            seed_prefix="demo-focus-v1",
        ),
        "piecewise": _curve(
            [
                {**row, "absolute_residual_depth_um": abs(row["residual_depth_um"])}
                for row in piecewise["corrected_rows"]
            ],
            design=PIECEWISE_NAME,
            depth_key="original_depth_um",
            value_key="absolute_residual_depth_um",
            depths=depths,
            seed_prefix="demo-focus-v1",
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    styles = {
        "exact_before": ("Exact · first frame", "exact", "--", "o"),
        "exact_after": ("Exact · corrected", "exact", "-", "o"),
        "stopped_after": ("Stopped · corrected", "stopped", "-", "s"),
        "piecewise_after": ("Piecewise · corrected", "piecewise", "-", "^"),
    }
    for key, (label, color_key, linestyle, marker) in styles.items():
        curve = pq_curves[key]
        x = np.asarray([row["depth_um"] for row in curve])
        mean = np.asarray([row["mean"] for row in curve])
        lower = np.asarray([row["ci_lower_95"] for row in curve])
        upper = np.asarray([row["ci_upper_95"] for row in curve])
        axes[0].fill_between(x, lower, upper, color=COLORS[color_key], alpha=0.10)
        axes[0].plot(
            x,
            mean,
            label=label,
            color=COLORS[color_key],
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=4.5,
        )
    axes[0].set_xlabel("Initial defocus (µm)")
    axes[0].set_ylabel("Panoptic quality (PQ; 0–1)")
    axes[0].set_title("Segmentation quality across depth")
    axes[0].set_xticks(depths)
    axes[0].legend(ncol=2, loc="lower center")
    add_panel_label(axes[0], "A")

    for system in ("exact", "stopped", "piecewise"):
        curve = focus_curves[system]
        x = np.asarray([row["depth_um"] for row in curve])
        mean = np.asarray([row["mean"] for row in curve])
        lower = np.asarray([row["ci_lower_95"] for row in curve])
        upper = np.asarray([row["ci_upper_95"] for row in curve])
        axes[1].fill_between(x, lower, upper, color=COLORS[system], alpha=0.10)
        axes[1].plot(
            x,
            mean,
            label=LABELS[system],
            color=COLORS[system],
            marker=MARKERS[system],
            linewidth=1.8,
            markersize=4.5,
        )
    axes[1].axhline(1.0, color=COLORS["muted"], linestyle=":", linewidth=1.2)
    axes[1].text(5.9, 1.04, "1 µm target", ha="right", va="bottom", color=COLORS["muted"])
    axes[1].set_xlabel("Initial defocus (µm)")
    axes[1].set_ylabel("Mean absolute residual defocus (µm)")
    axes[1].set_title("Autofocus residual across depth")
    axes[1].set_xticks(depths)
    axes[1].legend(loc="upper center", ncol=3)
    add_panel_label(axes[1], "B")
    fig.suptitle(
        "Closed-loop behavior on 27 hard-density validation wells",
        fontsize=14,
        fontweight="semibold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Lines are well means; bands are deterministic 95% well-bootstrap "
        "intervals (2,000 replicates).",
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=2.5)
    caption = (
        "Depth-resolved validation behavior over 27 hard-density wells. Corrected PQ remains "
        "near 0.55 across exact, stopped, and piecewise systems while exact autofocus retains "
        "lower residual defocus than its stopped-gradient control. Bands are grouped well "
        "bootstrap intervals and do not treat frames as independent replicates."
    )
    alt = (
        "Two line charts across six nonzero defocus depths. The left chart shows first-frame "
        "and corrected panoptic quality. The right chart shows absolute residual defocus for "
        "exact, stopped, and piecewise systems, with 95 percent well-bootstrap bands."
    )
    details = {
        "caption": caption,
        "alt": alt,
        "pq_curves": pq_curves,
        "focus_curves": focus_curves,
        "source_rows": {
            "exact_and_stopped": {
                "path": str(V2_4_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(V2_4_PATH),
            },
            "piecewise": {
                "path": str(PIECEWISE_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(PIECEWISE_PATH),
            },
        },
    }
    return _save_figure(fig, "pq-focus-depth", alt), details


def generate_causal_comparison(v2_4: dict) -> tuple[dict, dict]:
    name = EXACT_NAME
    paired = v2_4["paired_evidence"][name]
    stage = v2_4["stage_correction"][name]
    comparisons = [
        {
            "label": "Exact − stopped gradient",
            "reference": "forward-identical causal control",
            **paired["corrected_vs_stopped"],
            "status": "supported",
        },
        {
            "label": "Exact − piecewise-028",
            "reference": "strong noncausal design baseline",
            **paired["corrected_vs_piecewise"],
            "status": "not significant",
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [0.9, 1.35]})
    before = stage["hard_off_focus_pq_before"]
    after = stage["hard_off_focus_pq_after"]
    axes[0].plot([0, 1], [before, after], color=COLORS["exact"], linewidth=2.5, marker="o")
    axes[0].text(0, before - 0.006, f"{before:.4f}", ha="center", va="top")
    axes[0].text(1, after + 0.006, f"{after:.4f}", ha="center", va="bottom")
    axes[0].text(
        0.5,
        (before + after) / 2,
        f"+{after - before:.4f} PQ",
        ha="center",
        va="center",
        color=COLORS["exact"],
        fontweight="semibold",
    )
    axes[0].set_xticks([0, 1], ["First frame", "After stage action"])
    axes[0].set_ylabel("Panoptic quality (PQ; 0–1)")
    axes[0].set_ylim(0.47, 0.57)
    axes[0].set_title("Closed loop improves the exact design")
    axes[0].text(
        0.5,
        0.475,
        f"{stage['fraction_hard_frames_improved']:.1%} of hard frames improved",
        ha="center",
        va="bottom",
        color=COLORS["muted"],
    )
    add_panel_label(axes[0], "A")

    y = np.asarray([1, 0])
    estimates = np.asarray([row["mean_difference"] for row in comparisons])
    lower = np.asarray([row["ci_lower_95"] for row in comparisons])
    upper = np.asarray([row["ci_upper_95"] for row in comparisons])
    colors = [COLORS["exact"], COLORS["piecewise"]]
    markers = ["D", "o"]
    for index, _row in enumerate(comparisons):
        axes[1].errorbar(
            estimates[index],
            y[index],
            xerr=[[estimates[index] - lower[index]], [upper[index] - estimates[index]]],
            color=colors[index],
            marker=markers[index],
            markersize=7,
            linewidth=2,
            capsize=4,
        )
        axes[1].text(
            upper[index] + 0.00045,
            y[index],
            f"{estimates[index]:+.4f} [{lower[index]:+.4f}, {upper[index]:+.4f}]",
            va="center",
            fontsize=8,
        )
    axes[1].axvline(0.0, color=COLORS["ink"], linewidth=1.0)
    axes[1].axvline(0.005, color=COLORS["muted"], linewidth=1.1, linestyle=":")
    axes[1].set_yticks(y, [row["label"] for row in comparisons])
    axes[1].set_ylim(-0.7, 1.7)
    axes[1].set_xlim(-0.004, 0.0135)
    axes[1].set_xlabel("Paired corrected-PQ difference")
    axes[1].set_title("Causal evidence and the honest limitation")
    axes[1].text(
        0.0052,
        -0.57,
        "+0.005 target",
        ha="left",
        va="bottom",
        color=COLORS["muted"],
        fontsize=8,
    )
    add_panel_label(axes[1], "B")
    fig.suptitle(
        "Exact gradients improve closed-loop correction over a stopped-gradient control",
        fontsize=13.5,
        fontweight="semibold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "27 hard-density validation wells · paired well bootstrap · 2,000 replicates · "
        "PQ is not percent accuracy",
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=2.8)
    caption = (
        "The exact design increases its own hard off-focus PQ from 0.4922 to 0.5523 after "
        "one predicted stage action. The causal comparison against its forward-identical "
        "stopped-gradient control is +0.0063 PQ with a 95% well-bootstrap interval of +0.0007 "
        "to +0.0115. The comparison against piecewise-028 is only +0.0013 with an interval "
        "from −0.0017 to +0.0048, so superiority over piecewise is not supported."
    )
    alt = (
        "A paired chart shows exact-design PQ rising from 0.4922 before correction to 0.5523 "
        "after correction. A forest plot shows a positive confidence interval for exact versus "
        "stopped gradients and a confidence interval crossing zero for exact versus piecewise."
    )
    details = {
        "caption": caption,
        "alt": alt,
        "before_pq": before,
        "after_pq": after,
        "improved_fraction": stage["fraction_hard_frames_improved"],
        "comparisons": comparisons,
        "source": {
            "path": str(V2_4_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(V2_4_PATH),
            "pointers": [
                f"/stage_correction/{name}",
                f"/paired_evidence/{name}/corrected_vs_stopped",
                f"/paired_evidence/{name}/corrected_vs_piecewise",
            ],
        },
    }
    return _save_figure(fig, "causal-comparison", alt), details


def _box(axis: plt.Axes, x: float, y: float, width: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - 0.09),
        width,
        0.18,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor="white",
        edgecolor=color,
        linewidth=1.8,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=8.5, fontweight="semibold")


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.6,
            shrinkA=2,
            shrinkB=2,
            **kwargs,
        )
    )


def generate_architecture(derivative: dict) -> tuple[dict, dict]:
    fig, axis = plt.subplots(figsize=(13.4, 5.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.grid(False)
    xs = [0.08, 0.23, 0.38, 0.53, 0.68, 0.83, 0.94]
    widths = [0.115, 0.14, 0.12, 0.145, 0.11, 0.13, 0.08]
    labels = [
        "B7 phase\nmask",
        "JAX / Chromatix\noptics",
        "Depth image\nstack",
        "NumPy / SciPy\nautofocus",
        "Clipped stage\naction",
        "PyTorch / InstanSeg\ncorrected frame",
        "Task\nloss",
    ]
    colors = [
        COLORS["exact"],
        COLORS["exact"],
        COLORS["muted"],
        COLORS["piecewise"],
        COLORS["piecewise"],
        COLORS["stopped"],
        COLORS["stopped"],
    ]
    for x, width, label, color in zip(xs, widths, labels, colors, strict=True):
        _box(axis, x, 0.62, width, label, color)
    for left, left_width, right, right_width in zip(
        xs[:-1], widths[:-1], xs[1:], widths[1:], strict=True
    ):
        _arrow(
            axis,
            (left + left_width / 2, 0.62),
            (right - right_width / 2, 0.62),
            color=COLORS["ink"],
        )
    axis.text(0.02, 0.62, "Forward", rotation=90, va="center", ha="center", fontweight="semibold")

    _arrow(axis, (0.94, 0.42), (0.09, 0.42), color=COLORS["exact"])
    axis.text(
        0.51,
        0.36,
        "Exact reverse path crosses observer, corrected image, stage action, autofocus, and optics",
        ha="center",
        va="top",
        color=COLORS["exact"],
        fontweight="semibold",
    )
    axis.text(
        0.51,
        0.28,
        "Full-loop derivative gate: median relative error "
        f"{derivative['overall_median_relative_error']:.4f}; cosine "
        f"{derivative['overall_cosine_agreement']:.5f}",
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
    )

    _arrow(axis, (0.94, 0.15), (0.58, 0.15), color=COLORS["stopped"], linestyle="--")
    axis.plot([0.55, 0.55], [0.10, 0.20], color=COLORS["stopped"], linewidth=3)
    axis.text(
        0.55,
        0.075,
        "stop-gradient at stage path",
        ha="center",
        va="top",
        color=COLORS["stopped"],
        fontsize=8,
    )
    axis.text(0.02, 0.15, "Control", rotation=90, va="center", ha="center", fontweight="semibold")
    axis.text(
        0.75,
        0.09,
        "Same forward computation; one causal learning path is removed",
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
    )
    fig.suptitle(
        "TessScope closes the gradient loop across three scientific runtimes",
        fontsize=14,
        fontweight="semibold",
        y=0.97,
    )
    caption = (
        "TessScope forms a depth stack with JAX/Chromatix, predicts defocus with a frozen "
        "NumPy/SciPy ridge autofocus model, applies a clipped stage action, and evaluates the "
        "corrected frame with frozen PyTorch/InstanSeg. Exact reverse-mode gradients cross the "
        "stage-action path; the stopped-gradient control keeps the forward computation identical "
        "but removes that causal learning signal."
    )
    alt = (
        "A left-to-right pipeline connects a B7 phase mask, JAX Chromatix optics, depth image "
        "stack, NumPy SciPy autofocus, stage action, PyTorch InstanSeg corrected frame, and task "
        "loss. A blue reverse arrow spans the full exact path; a dashed orange control arrow "
        "stops at the stage-action boundary."
    )
    details = {
        "caption": caption,
        "alt": alt,
        "derivative_gate": {
            "overall_median_relative_error": derivative["overall_median_relative_error"],
            "overall_cosine_agreement": derivative["overall_cosine_agreement"],
            "forward_parity_absolute": derivative["forward_parity_absolute"],
            "passed": derivative["passed"],
            "source": {
                "path": str(DERIVATIVE_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(DERIVATIVE_PATH),
                "pointers": [
                    "/overall_median_relative_error",
                    "/overall_cosine_agreement",
                    "/forward_parity_absolute",
                    "/passed",
                ],
            },
        },
    }
    return _save_figure(fig, "gradient-architecture", alt), details


def _replay_figure(
    arrays: dict[str, np.ndarray], metadata: dict, depth_index: int
) -> tuple[plt.Figure, list[dict]]:
    depth = float(metadata["depths_um"][depth_index])
    scale = float(metadata["observer"]["transform"]["scale"])
    target = arrays["target_labels"]
    system_index = {name: index for index, name in enumerate(metadata["system_order"])}
    corrected_index = {
        name: index for index, name in enumerate(metadata["corrected_system_order"])
    }
    fig, axes = plt.subplots(3, 2, figsize=(6.7, 8.7))
    traces = []
    for row, system in enumerate(metadata["corrected_system_order"]):
        trace = _frame(metadata, system, depth)
        traces.append(trace)
        _draw_overlay_panel(
            axes[row, 0],
            arrays["sensor_before"][system_index[system], depth_index],
            target,
            arrays["labels_before"][system_index[system], depth_index],
            scale=scale,
            title=f"{LABELS[system]} · first",
            subtitle=f"PQ {trace['before_pq']:.3f}",
        )
        after = f"PQ {trace['after_pq']:.3f}" if "after_pq" in trace else "focus plane · not scored"
        _draw_overlay_panel(
            axes[row, 1],
            arrays["sensor_corrected"][corrected_index[system], depth_index],
            target,
            arrays["labels_corrected"][corrected_index[system], depth_index],
            scale=scale,
            title=f"{LABELS[system]} · corrected",
            subtitle=f"residual {trace['residual_depth_um']:+.2f} µm · {after}",
        )
    fig.suptitle(
        f"Validation replay · initial defocus {format_depth(depth)}",
        fontsize=13,
        fontweight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "n21_s1 · cached deterministic evidence · identical global display transform",
        ha="center",
        color=COLORS["muted"],
        fontsize=8,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.055, wspace=0.10, hspace=0.24)
    return fig, traces


def generate_replay(arrays: dict[str, np.ndarray], metadata: dict) -> tuple[dict, list[dict]]:
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    frame_manifest = []
    frame_paths = []
    for index, depth in enumerate(metadata["depths_um"]):
        fig, traces = _replay_figure(arrays, metadata, index)
        path = REPLAY_ROOT / f"frame-{index:02d}.png"
        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "TessScope", "Description": f"Validation replay {depth:+.1f} µm"},
        )
        plt.close(fig)
        frame_paths.append(path)
        frame_manifest.append(
            {
                "index": index,
                "depth_um": depth,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(path),
                "size_bytes": path.stat().st_size,
                "traced_frames": traces,
            }
        )

    images = [Image.open(path).convert("RGB") for path in frame_paths]
    gif_path = REPLAY_ROOT / "validation-depth-sweep.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for image in images:
        image.close()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the broadly playable MP4 output")
    mp4_path = REPLAY_ROOT / "validation-depth-sweep.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1.25",
            "-i",
            str(REPLAY_ROOT / "frame-%02d.png"),
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ],
        check=True,
    )
    outputs = {
        "frames": frame_manifest,
        "gif": {
            "path": str(gif_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(gif_path),
            "size_bytes": gif_path.stat().st_size,
            "duration_ms_per_frame": 800,
        },
        "mp4": {
            "path": str(mp4_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(mp4_path),
            "size_bytes": mp4_path.stat().st_size,
            "frames_per_second": 1.25,
            "codec": "H.264",
        },
    }
    return outputs, frame_manifest


def _write_captions(figures: dict, replay: dict) -> None:
    lines = [
        "# TessScope demo captions and alt text",
        "",
        "All evidence is validation-only. PQ is panoptic quality, not percent accuracy. "
        "The locked test remains sealed, and no physical microscope has validated the system.",
        "",
    ]
    for key in (
        "causal-comparison",
        "matched-microscopy",
        "pq-focus-depth",
        "pupil-psf-depth",
        "gradient-architecture",
    ):
        lines.extend(
            [
                f"## {key.replace('-', ' ').title()}",
                "",
                f"**Caption:** {figures[key]['caption']}",
                "",
                f"**Alt text:** {figures[key]['alt']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Validation depth sweep",
            "",
            "**Caption:** Cached replay of the frozen representative validation field across "
            "the complete seven-depth grid. Exact, stopped-gradient, and piecewise systems "
            "use identical display and label-overlay rules before and after one stage action.",
            "",
            "**Alt text:** An animation steps from minus six to plus six micrometres of initial "
            "defocus. Three rows compare exact, stopped-gradient, and piecewise first and "
            "corrected frames with automated-reference and InstanSeg boundaries.",
            "",
            f"MP4: `{replay['mp4']['path']}`",
            "",
            f"GIF: `{replay['gif']['path']}`",
            "",
        ]
    )
    CAPTIONS_PATH.write_text("\n".join(lines))


def main() -> None:
    apply_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    replay_metadata = _json(REPLAY_METADATA_PATH)
    if replay_metadata["split"] != "validation" or replay_metadata["test_accessed"]:
        raise ValueError("Demo rendering requires a validation-only sealed replay")
    with np.load(REPLAY_PATH, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    v2_4 = _json(V2_4_PATH)
    piecewise = _json(PIECEWISE_PATH)
    derivative = _json(DERIVATIVE_PATH)

    figures = {}
    files, details = generate_matched_microscopy(arrays, replay_metadata)
    figures["matched-microscopy"] = {**details, "files": files}
    files, details = generate_pupil_psf(arrays, replay_metadata)
    figures["pupil-psf-depth"] = {**details, "files": files}
    files, details = generate_depth_curves(v2_4, piecewise)
    figures["pq-focus-depth"] = {**details, "files": files}
    files, details = generate_causal_comparison(v2_4)
    figures["causal-comparison"] = {**details, "files": files}
    files, details = generate_architecture(derivative)
    figures["gradient-architecture"] = {**details, "files": files}
    replay, _ = generate_replay(arrays, replay_metadata)
    _write_captions(figures, replay)

    manifest = {
        "schema_version": 1,
        "status": "complete_validation_only_publication_outputs",
        "test_accessed": False,
        "display_integrity": {
            "sensor_transform": replay_metadata["observer"]["transform"],
            "geometry": "same selected field, crop, sensor grid, and nearest label interpolation",
            "per_image_normalization": False,
            "scale_bar": "omitted because authoritative display-plane calibration is absent",
        },
        "inputs": {
            "replay_arrays": {
                "path": str(REPLAY_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(REPLAY_PATH),
            },
            "replay_metadata": {
                "path": str(REPLAY_METADATA_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(REPLAY_METADATA_PATH),
            },
            "claim_matrix": {
                "path": str(CLAIMS_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(CLAIMS_PATH),
            },
            "visual_contract": {
                "path": str(VISUAL_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(VISUAL_PATH),
            },
        },
        "figures": figures,
        "replay": replay,
        "captions": {
            "path": str(CAPTIONS_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(CAPTIONS_PATH),
        },
        "limitations": [
            "PQ is panoptic quality, not percent accuracy.",
            "All displayed evaluation evidence is validation-only.",
            "The locked BBBC006 test remains sealed.",
            "No physical microscope has validated this system.",
            "The animation replays cached evidence and performs no live inference.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"figures={len(figures)}")
    print(f"replay_frames={len(replay['frames'])}")
    print(f"manifest={MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"manifest_sha256={sha256_path(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
