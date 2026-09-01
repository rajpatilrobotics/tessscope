"""Audit the frozen v2 near-miss using training/validation evidence only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch

from tessscope.evaluation.metrics import instance_metrics, valid_region_labels
from tessscope.evaluation.route import segment_sensor_batch
from tessscope.observer.instanseg import load_frozen_model
from tessscope.observer.loss import (
    PHASE_L2_WEIGHT,
    ObserverTransform,
    aggregate_depth_loss,
    design_loss_components,
)
from tessscope.optics.model import phase_coefficients
from tessscope.v2.optics.model import simulate_cubic_rate, simulate_noisy_sensor
from tessscope.v2.optimization.served import (
    DEPTHS,
    V2SystemCalibration,
    collect_patches,
)
from tessscope.v2_1.audit import (
    correlation_table,
    group_summary,
    paired_comparison,
    summarize_hard_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "v2"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "v2_1" / "audit"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "v2_1"

BASE_HARD = V2_ROOT / "gate4" / "hard-validation.json"
REFINEMENT_SOURCES = (
    (
        "seg-start",
        V2_ROOT / "gate4" / "hard-validation-refinement.json",
        V2_ROOT / "optimization" / "pareto-refinement.json",
    ),
    (
        "joint-start",
        V2_ROOT / "gate4" / "hard-validation-refinement-from-joint.json",
        V2_ROOT / "optimization" / "pareto-refinement-from-joint.json",
    ),
    (
        "projected",
        V2_ROOT / "gate4" / "hard-validation-projected-gradients.json",
        V2_ROOT / "optimization" / "projected-gradient-candidates.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--component-batch-size", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _copy_rows(rows: list[dict], candidate_id: str, name_key: str) -> list[dict]:
    return [
        {
            **row,
            "original_name": row[name_key],
            "candidate_id": candidate_id,
            "density_group": "dense" if row["hard_dense_patch"] else "sparse",
        }
        for row in rows
    ]


def candidate_registry() -> list[dict]:
    """Join hard rows, optics parameters, and original soft-validation records."""
    base = load_json(BASE_HARD)
    optimization = load_json(V2_ROOT / "optimization" / "summary.json")
    designs = optimization["designs"]
    selected_joint = next(
        row
        for row in designs["exact_joint_starts"]
        if row["name"] == designs["selected_exact_joint"]
    )
    soft_records = {
        "clear": designs["clear"],
        "astigmatic": designs["classical_astigmatic"],
        "segmentation_only": designs["segmentation_only"],
        "focus_only": designs["focus_only"],
        "exact_joint": selected_joint,
        "broken_focus_gradient": designs["broken_focus_gradient"],
        "naive_superposition": designs["naive_superposition"],
        "derivative_free": designs["derivative_free"],
    }
    entries = []
    for name, design in base["design_registry"].items():
        entries.append(
            {
                "candidate_id": name,
                "label": name.replace("_", " "),
                "family": design["family"],
                "parameters": design.get("parameters"),
                "strength": design.get("strength"),
                "source_group": "matched v2",
                "hard_rows": _copy_rows(
                    [row for row in base["rows"] if row["design"] == name],
                    name,
                    "design",
                ),
                "stored_corrected_rows": [
                    {**row, "candidate_id": name}
                    for row in base["corrected_rows"]
                    if name == "exact_joint"
                ],
                "original_soft_validation": soft_records.get(name, {}).get("validation"),
            }
        )
    for prefix, hard_path, optimization_path in REFINEMENT_SOURCES:
        hard = load_json(hard_path)
        runs = load_json(optimization_path)["candidates"]
        run_by_name = {row["name"]: row for row in runs}
        for original_name in hard["summaries"]:
            run = run_by_name[original_name]
            candidate_id = f"{prefix}/{original_name}"
            entries.append(
                {
                    "candidate_id": candidate_id,
                    "label": candidate_id.replace("joint-refine-focus-", "focus "),
                    "family": "zernike",
                    "parameters": run["final_parameters"],
                    "strength": None,
                    "source_group": prefix,
                    "hard_rows": _copy_rows(
                        [row for row in hard["rows"] if row["candidate"] == original_name],
                        candidate_id,
                        "candidate",
                    ),
                    "stored_corrected_rows": [
                        {**row, "candidate_id": candidate_id}
                        for row in hard["corrected_rows"]
                        if row["candidate"] == original_name
                    ],
                    "original_soft_validation": run.get("validation"),
                }
            )
    return entries


def simulate_entry(
    entry: dict,
    objects: np.ndarray,
    depths: np.ndarray,
    calibration: V2SystemCalibration,
) -> np.ndarray:
    if entry["family"] == "cubic":
        sensor = simulate_cubic_rate(
            jnp.asarray(entry["strength"], dtype=jnp.float32),
            jnp.asarray(objects),
            jnp.asarray(depths),
            exposure_gain=calibration.exposure_gain,
            depth_scale=calibration.depth_scale,
            axial_offset_um=calibration.axial_offset_um,
        )
    else:
        noise = np.zeros((len(objects), len(depths), 256, 256), dtype=np.float32)
        sensor = simulate_noisy_sensor(
            jnp.asarray(entry["parameters"], dtype=jnp.float32),
            jnp.asarray(objects),
            jnp.asarray(depths),
            jnp.asarray(noise),
            expected_photons=calibration.expected_photons,
            exposure_gain=calibration.exposure_gain,
            depth_scale=calibration.depth_scale,
            axial_offset_um=calibration.axial_offset_um,
        )
    return np.maximum(np.asarray(sensor), 0.0)


def soft_loss_components(
    entry: dict,
    patches: list,
    calibration: V2SystemCalibration,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict:
    transform = ObserverTransform(mode="affine", offset=0.0, scale=calibration.exposure_gain)
    totals = {"seed": 0.0, "instance": 0.0}
    per_depth = np.zeros(len(DEPTHS), dtype=np.float64)
    count = 0
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            chunk = patches[start : start + batch_size]
            sensor = simulate_entry(
                entry,
                np.stack([patch.object_image for patch in chunk]).astype(np.float32),
                DEPTHS,
                calibration,
            )
            components = design_loss_components(
                model,
                torch.as_tensor(sensor, device=device),
                torch.as_tensor(
                    np.stack([patch.instance_labels for patch in chunk]),
                    device=device,
                    dtype=torch.long,
                ),
                torch.as_tensor(
                    np.stack([patch.centers_yx for patch in chunk]),
                    device=device,
                    dtype=torch.float32,
                ),
                torch.as_tensor(
                    np.stack([patch.valid_objects for patch in chunk]),
                    device=device,
                    dtype=torch.bool,
                ),
                transform,
            )
            chunk_count = len(chunk)
            totals["seed"] += float(components["seed"].cpu()) * chunk_count
            totals["instance"] += float(components["instance"].cpu()) * chunk_count
            per_depth += np.asarray(components["per_depth"].cpu()) * chunk_count
            count += chunk_count
    mean_per_depth = per_depth / count
    if entry["family"] == "cubic":
        phase_rms = abs(float(entry["strength"]))
    else:
        coefficients = np.asarray(
            phase_coefficients(jnp.asarray(entry["parameters"])), dtype=np.float64
        )
        phase_rms = float(np.linalg.norm(coefficients))
    aggregated = float(aggregate_depth_loss(torch.as_tensor(mean_per_depth)))
    return {
        "seed_loss": totals["seed"] / count,
        "instance_loss": totals["instance"] / count,
        "task_loss": aggregated + PHASE_L2_WEIGHT * phase_rms**2,
        "task_loss_without_phase_penalty": aggregated,
        "phase_rms_radians": phase_rms,
        "per_depth_loss": mean_per_depth.tolist(),
    }


def recompute_correction(
    entry: dict,
    patches: list,
    calibration: V2SystemCalibration,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    transform = ObserverTransform(mode="affine", offset=0.0, scale=calibration.exposure_gain)
    rows_by_field = {}
    for row in entry["hard_rows"]:
        rows_by_field.setdefault(row["field_id"], []).append(row)
    corrected = []
    for patch in patches:
        original_rows = sorted(rows_by_field[patch.field_id], key=lambda row: row["depth_um"])
        residual_depths = np.asarray(
            [row["depth_um"] - row["predicted_depth_um"] for row in original_rows],
            dtype=np.float32,
        )
        sensor = simulate_entry(
            entry,
            patch.object_image[None].astype(np.float32),
            residual_depths,
            calibration,
        )[0]
        predictions = segment_sensor_batch(
            model,
            sensor,
            patch.instance_labels.shape,
            device=device,
            maximum_batch_size=4,
            transform=transform,
        )
        for original, residual, prediction in zip(
            original_rows, residual_depths, predictions, strict=True
        ):
            target, valid_prediction = valid_region_labels(
                patch.instance_labels, prediction, margin=62
            )
            corrected.append(
                {
                    "candidate_id": entry["candidate_id"],
                    "well": patch.well,
                    "field_id": patch.field_id,
                    "original_depth_um": float(original["depth_um"]),
                    "residual_depth_um": float(residual),
                    "hard_dense_patch": bool(original["hard_dense_patch"]),
                    "before_pq": float(original["panoptic_quality"]),
                    "after_pq": instance_metrics(target, valid_prediction).panoptic_quality,
                }
            )
    off_focus = [row for row in corrected if row["original_depth_um"] != 0.0]
    hard = [row for row in off_focus if row["hard_dense_patch"]]
    return {
        "rows": corrected,
        "all_off_focus": {
            "mean_absolute_depth_before_um": float(
                np.mean([abs(row["original_depth_um"]) for row in off_focus])
            ),
            "mean_absolute_depth_after_um": float(
                np.mean([abs(row["residual_depth_um"]) for row in off_focus])
            ),
            "pq_before": float(np.mean([row["before_pq"] for row in off_focus])),
            "pq_after": float(np.mean([row["after_pq"] for row in off_focus])),
        },
        "hard_dense_off_focus": {
            "mean_absolute_depth_before_um": float(
                np.mean([abs(row["original_depth_um"]) for row in hard])
            ),
            "mean_absolute_depth_after_um": float(
                np.mean([abs(row["residual_depth_um"]) for row in hard])
            ),
            "pq_before": float(np.mean([row["before_pq"] for row in hard])),
            "pq_after": float(np.mean([row["after_pq"] for row in hard])),
        },
    }


def leave_one_field_out(rows: list[dict]) -> list[dict]:
    fields = sorted({str(row["field_id"]) for row in rows})
    output = []
    for field in fields:
        summary = summarize_hard_rows([row for row in rows if str(row["field_id"]) != field])
        output.append(
            {
                "omitted_field_id": field,
                "hard_dense_off_focus_pq": summary["hard_dense_off_focus"]["panoptic_quality"],
                "focus_mae_um": summary["focus"]["mae_um"],
                "signed_direction_accuracy": summary["focus"]["signed_direction_accuracy"],
            }
        )
    return output


def plot_pareto(candidates: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    groups = sorted({row["source_group"] for row in candidates})
    colors = plt.get_cmap("tab10")
    for index, group in enumerate(groups):
        rows = [row for row in candidates if row["source_group"] == group]
        axis.scatter(
            [row["focus_mae_um"] for row in rows],
            [row["hard_dense_off_focus_pq"] for row in rows],
            s=55,
            label=group,
            color=colors(index),
            alpha=0.85,
        )
        annotated = {
            "clear",
            "astigmatic",
            "segmentation_only",
            "focus_only",
            "exact_joint",
            "naive_superposition",
            "derivative_free",
            "joint-start/joint-refine-focus-0.03",
            "joint-start/joint-refine-focus-0.05",
            "joint-start/joint-refine-focus-0.07",
            "projected/projected-exact-joint-0.25",
        }
        for row in rows:
            if row["candidate_id"] not in annotated:
                continue
            axis.annotate(
                row["label"],
                (row["focus_mae_um"], row["hard_dense_off_focus_pq"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    by_id = {row["candidate_id"]: row for row in candidates}
    segmentation_floor = by_id["segmentation_only"]["hard_dense_off_focus_pq"] - 0.01
    clear_floor = by_id["clear"]["hard_dense_off_focus_pq"] + 0.01
    superposition_mae = by_id["naive_superposition"]["focus_mae_um"]
    axis.axhline(segmentation_floor, color="black", linestyle="--", linewidth=1)
    axis.axhline(clear_floor, color="gray", linestyle=":", linewidth=1)
    axis.axvline(superposition_mae, color="black", linestyle=":", linewidth=1)
    axis.set_xlabel("Signed focus MAE (µm; lower is better)")
    axis.set_ylabel("Hard dense off-focus PQ (higher is better)")
    axis.set_title("TessScope v2 near-miss Pareto audit (validation only)")
    axis.legend(ncols=2, fontsize=8)
    axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_breakdown(comparisons: dict, output: Path) -> None:
    depth = comparisons["by_depth"]
    density = comparisons["by_density"]
    well = comparisons["by_well"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].bar(
        [float(row["depth_um"]) for row in depth],
        [row["delta_panoptic_quality"] for row in depth],
    )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("Defocus depth (µm)")
    axes[0].set_ylabel("PQ delta: 0.03 near-miss − superposition")
    axes[0].set_title("Paired depth effect")

    density_x = np.arange(len(density))
    width = 0.36
    axes[1].bar(
        density_x - width / 2,
        [row["delta_panoptic_quality"] for row in density],
        width,
        label="PQ delta",
    )
    axes[1].bar(
        density_x + width / 2,
        [row["delta_recognition_quality"] for row in density],
        width,
        label="RQ delta",
    )
    axes[1].set_xticks(density_x, [row["density_group"] for row in density])
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Candidate − superposition")
    axes[1].set_title("Sparse versus dense fields")
    axes[1].legend(fontsize=8)

    axes[2].bar(
        [row["well"] for row in well],
        [row["delta_panoptic_quality"] for row in well],
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].tick_params(axis="x", rotation=60)
    axes[2].set_xlabel("Validation well")
    axes[2].set_ylabel("Paired PQ delta")
    axes[2].set_title("Well influence")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_table(candidates: list[dict], output: Path) -> None:
    columns = (
        "candidate_id",
        "source_group",
        "phase_rms_radians",
        "hard_dense_off_focus_pq",
        "hard_dense_recognition_quality",
        "hard_dense_segmentation_quality",
        "hard_dense_foreground_dice",
        "hard_dense_absolute_count_error",
        "focus_mae_um",
        "signed_direction_accuracy",
        "seed_loss",
        "instance_loss",
        "task_loss",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in candidates)


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / "near-miss.json"
    table = OUTPUT_ROOT / "candidate-table.csv"
    pareto_figure = FIGURE_ROOT / "near-miss-pareto.png"
    breakdown_figure = FIGURE_ROOT / "near-miss-breakdown.png"
    for target in (output, table, pareto_figure, breakdown_figure):
        if target.exists() and not args.force:
            raise SystemExit(f"Audit output already exists: {target}")

    calibration = V2SystemCalibration.load()
    patches = collect_patches("validation", 12, seed=53)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    entries = candidate_registry()
    candidate_rows = []
    for index, entry in enumerate(entries, start=1):
        hard = summarize_hard_rows(entry["hard_rows"])
        soft = soft_loss_components(
            entry,
            patches,
            calibration,
            model,
            device,
            args.component_batch_size,
        )
        dense = hard["hard_dense_off_focus"]
        candidate_rows.append(
            {
                "candidate_id": entry["candidate_id"],
                "label": entry["label"],
                "source_group": entry["source_group"],
                "phase_rms_radians": soft["phase_rms_radians"],
                "hard_dense_off_focus_pq": dense["panoptic_quality"],
                "hard_dense_recognition_quality": dense["recognition_quality"],
                "hard_dense_segmentation_quality": dense["segmentation_quality"],
                "hard_dense_foreground_dice": dense["foreground_dice"],
                "hard_dense_absolute_count_error": dense["absolute_count_error"],
                "mean_off_focus_pq": hard["all_off_focus"]["panoptic_quality"],
                "focus_mae_um": hard["focus"]["mae_um"],
                "nonzero_focus_mae_um": hard["focus"]["nonzero_mae_um"],
                "signed_direction_accuracy": hard["focus"]["signed_direction_accuracy"],
                **soft,
                "original_soft_validation": entry["original_soft_validation"],
            }
        )
        print(
            json.dumps(
                {
                    "candidate": entry["candidate_id"],
                    "progress": f"{index}/{len(entries)}",
                    "hard_pq": dense["panoptic_quality"],
                    "focus_mae_um": hard["focus"]["mae_um"],
                    "task_loss": soft["task_loss"],
                }
            ),
            flush=True,
        )

    by_id = {entry["candidate_id"]: entry for entry in entries}
    near_id = "joint-start/joint-refine-focus-0.03"
    super_id = "naive_superposition"
    near = by_id[near_id]
    superposition = by_id[super_id]
    comparisons = {
        "by_well": paired_comparison(
            superposition["hard_rows"], near["hard_rows"], group_key="well"
        ),
        "by_depth": paired_comparison(
            superposition["hard_rows"], near["hard_rows"], group_key="depth_um"
        ),
        "by_density": paired_comparison(
            superposition["hard_rows"], near["hard_rows"], group_key="density_group"
        ),
        "near_miss_focus_by_depth": group_summary(near["hard_rows"], "depth_um"),
        "superposition_focus_by_depth": group_summary(superposition["hard_rows"], "depth_um"),
        "near_miss_leave_one_field_out": leave_one_field_out(near["hard_rows"]),
    }
    corrections = {
        near_id: recompute_correction(near, patches, calibration, model, device),
        super_id: recompute_correction(superposition, patches, calibration, model, device),
    }
    correlations = correlation_table(
        candidate_rows,
        predictor_keys=("seed_loss", "instance_loss", "task_loss"),
        outcome_keys=(
            "hard_dense_off_focus_pq",
            "hard_dense_recognition_quality",
            "hard_dense_segmentation_quality",
            "hard_dense_foreground_dice",
            "hard_dense_absolute_count_error",
        ),
    )
    report = {
        "status": "complete_validation_only_causal_audit_part_1",
        "test_accessed": False,
        "candidate_count": len(candidate_rows),
        "validation_fields": [patch.field_id for patch in patches],
        "candidate_table": candidate_rows,
        "near_miss_vs_superposition": comparisons,
        "stage_correction": corrections,
        "soft_to_hard_correlations": correlations,
        "provisional_diagnosis": {
            "basis_expressivity": "not identifiable from B6-only evidence",
            "optimization": "requires exact branch-gradient path audit",
            "soft_hard_alignment": "quantified here; interpret correlation signs and magnitudes",
            "validation_noise": (
                "leave-one-field-out sensitivity quantified; expand by well before promotion"
            ),
        },
        "figures": [
            str(pareto_figure.relative_to(PROJECT_ROOT)),
            str(breakdown_figure.relative_to(PROJECT_ROOT)),
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_table(candidate_rows, table)
    plot_pareto(candidate_rows, pareto_figure)
    plot_breakdown(comparisons, breakdown_figure)
    print(
        json.dumps(
            {
                "output": str(output),
                "table": str(table),
                "figures": [str(pareto_figure), str(breakdown_figure)],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
