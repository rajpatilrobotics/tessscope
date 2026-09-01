"""Evaluate the frozen v2.4 checkpoint pool on unchanged soft validation."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import DEPTHS, V2SystemCalibration, collect_patches
from tessscope.v2_1.optics.model import (
    phase_coefficients_b7,
    psf_support_energy_fraction_b7,
)
from tessscope.v2_3.closed_loop import (
    OBJECTIVE_PROFILES,
    closed_loop_value,
    materialize_closed_loop_batch,
)
from tessscope.v2_4.checkpoints import (
    FIRST_SEGMENTATION_MAXIMUM,
    eligible_intervals,
    is_soft_eligible,
    select_checkpoints,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_3"
    / "optimization"
    / "closed-loop-matrix.json"
)
POOL = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "preregistration"
    / "checkpoint-pool.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "checkpoint-audit.json"
)
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "v2_4" / "checkpoint-trajectories"
CACHE_ROOT = PROJECT_ROOT / "artifacts" / "runtime-runs" / "v2_4-checkpoint-audit"
SUPPORT_BATCH_SIZE = 8
ENDPOINT_REPRODUCTION_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8407")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def average_validation(services, parameters, batches, calibration) -> dict:
    rows = []
    for batch in batches:
        _, diagnostics = closed_loop_value(
            *services,
            parameters,
            batch,
            calibration,
            OBJECTIVE_PROFILES["balanced"],
            gradient_mode="exact",
        )
        rows.append(diagnostics)
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def objective_from_diagnostics(diagnostics: dict, profile: str) -> float:
    weights = OBJECTIVE_PROFILES[profile]
    return float(
        weights.first_segmentation * diagnostics["first_segmentation_loss"]
        + weights.final_segmentation * diagnostics["final_segmentation_loss"]
        + weights.normalized_residual_depth_squared
        * diagnostics["normalized_residual_depth_squared"]
        + weights.normalized_stage_action_squared
        * diagnostics["normalized_stage_action_squared"]
        + weights.fixed_second_exposure_cost
    )


def parameter_lookup(source: dict) -> dict[str, list[float]]:
    lookup = {}
    for run in source["exact_runs"]:
        for trace_row in run["trace"]:
            name = f"{run['name']}-step-{int(trace_row['step']):02d}"
            lookup[name] = trace_row["parameters"]
    return lookup


def minimum_support_by_hash(
    unique_hashes: list[str],
    parameters_by_hash: dict[str, list[float]],
) -> dict[str, float]:
    depths = jnp.asarray(DEPTHS, dtype=jnp.float32)

    @jax.jit
    def support_batch(parameters: jax.Array) -> jax.Array:
        def one_parameter(value):
            return jnp.min(
                jax.vmap(lambda depth: psf_support_energy_fraction_b7(value, depth))(
                    depths
                )
            )

        return jax.vmap(one_parameter)(parameters)

    output = {}
    for first in range(0, len(unique_hashes), SUPPORT_BATCH_SIZE):
        hashes = unique_hashes[first : first + SUPPORT_BATCH_SIZE]
        values = [parameters_by_hash[digest] for digest in hashes]
        if len(values) < SUPPORT_BATCH_SIZE:
            values.extend([values[-1]] * (SUPPORT_BATCH_SIZE - len(values)))
        supports = np.asarray(
            support_batch(jnp.asarray(values, dtype=jnp.float32))
        )[: len(hashes)]
        output.update(
            {digest: float(support) for digest, support in zip(hashes, supports, strict=True)}
        )
    return output


def render_trajectory_figures(rows: list[dict], source: dict, selected: set[str]) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    rows_by_run = defaultdict(list)
    for row in rows:
        rows_by_run[row["source_run"]].append(row)
    source_by_name = {run["name"]: run for run in source["exact_runs"]}
    for run_name, run_rows in rows_by_run.items():
        run_rows.sort(key=lambda row: row["step"])
        source_run = source_by_name[run_name]
        steps = [0, *[row["step"] for row in run_rows]]
        first_losses = [
            source_run["start_validation"]["first_segmentation_loss"],
            *[row["first_segmentation_loss"] for row in run_rows],
        ]
        residuals = [
            source_run["start_validation"]["residual_mae_um"],
            *[row["residual_mae_um"] for row in run_rows],
        ]
        figure, first_axis = plt.subplots(figsize=(8, 4.8))
        residual_axis = first_axis.twinx()
        first_axis.plot(steps, first_losses, color="#3155a4", marker="o", markersize=3)
        residual_axis.plot(steps, residuals, color="#d55e00", marker="s", markersize=3)
        first_axis.axhline(
            FIRST_SEGMENTATION_MAXIMUM,
            color="#3155a4",
            linestyle="--",
            linewidth=1,
            label="first-loss ceiling",
        )
        eligible = [row for row in run_rows if row["eligible"]]
        if eligible:
            first_axis.scatter(
                [row["step"] for row in eligible],
                [row["first_segmentation_loss"] for row in eligible],
                color="#009e73",
                marker="o",
                s=42,
                label="soft eligible",
                zorder=4,
            )
        promoted = [row for row in run_rows if row["name"] in selected]
        if promoted:
            first_axis.scatter(
                [row["step"] for row in promoted],
                [row["first_segmentation_loss"] for row in promoted],
                color="#cc79a7",
                marker="*",
                s=150,
                label="selected",
                zorder=5,
            )
        first_axis.set_xlabel("Adam checkpoint step (0 is frozen start control)")
        first_axis.set_ylabel("First-frame segmentation loss", color="#3155a4")
        residual_axis.set_ylabel("Residual-defocus MAE (µm)", color="#d55e00")
        first_axis.set_title(run_name)
        first_axis.grid(alpha=0.2)
        first_axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(FIGURE_ROOT / f"{run_name}.png", dpi=170)
        plt.close(figure)


def render_tradeoff_figure(
    rows: list[dict], selected: set[str], x_key: str, x_label: str, filename: str
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    colors = ["#009e73" if row["eligible"] else "#999999" for row in rows]
    axis.scatter(
        [row[x_key] for row in rows],
        [row["residual_mae_um"] for row in rows],
        c=colors,
        alpha=0.65,
        s=24,
    )
    promoted = [row for row in rows if row["name"] in selected]
    if promoted:
        axis.scatter(
            [row[x_key] for row in promoted],
            [row["residual_mae_um"] for row in promoted],
            color="#cc79a7",
            edgecolor="black",
            marker="*",
            s=180,
            label="selected",
        )
        axis.legend()
    if x_key == "first_segmentation_loss":
        axis.axvline(
            FIRST_SEGMENTATION_MAXIMUM,
            color="#3155a4",
            linestyle="--",
            linewidth=1,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("Residual-defocus MAE (µm)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_ROOT / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"V2.4 checkpoint audit already exists: {OUTPUT}")
    source = json.loads(SOURCE.read_text())
    pool = json.loads(POOL.read_text())
    parameters_by_name = parameter_lookup(source)
    run_by_name = {run["name"]: run for run in source["exact_runs"]}
    parameters_by_hash = {}
    records_by_hash = defaultdict(list)
    for record in pool["records"]:
        parameters = parameters_by_name[record["name"]]
        parameters_by_hash.setdefault(record["parameter_sha256"], parameters)
        records_by_hash[record["parameter_sha256"]].append(record)
    unique_hashes = list(parameters_by_hash)
    if len(unique_hashes) != pool["unique_parameter_hash_count"]:
        raise SystemExit("Unique checkpoint parameter count differs from frozen manifest")

    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    validation_patches = collect_patches("validation", 12, seed=53)
    validation_batches = [
        materialize_closed_loop_batch(
            validation_patches[index : index + 3], noise_seed=None
        )
        for index in range(0, len(validation_patches), 3)
    ]
    supports = minimum_support_by_hash(unique_hashes, parameters_by_hash)
    diagnostics_by_hash = {}
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for index, digest in enumerate(unique_hashes, start=1):
        cache_path = CACHE_ROOT / f"{digest}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
        else:
            started = time.perf_counter()
            diagnostics = average_validation(
                services,
                np.asarray(parameters_by_hash[digest], dtype=np.float32),
                validation_batches,
                calibration,
            )
            cached = {
                "parameter_sha256": digest,
                "elapsed_seconds": time.perf_counter() - started,
                "diagnostics": diagnostics,
            }
            cache_path.write_text(json.dumps(cached) + "\n")
        diagnostics_by_hash[digest] = cached
        diagnostics = cached["diagnostics"]
        print(
            json.dumps(
                {
                    "unique_checkpoint": index,
                    "of": len(unique_hashes),
                    "representative": records_by_hash[digest][0]["name"],
                    "first_loss": round(diagnostics["first_segmentation_loss"], 6),
                    "residual_mae_um": round(diagnostics["residual_mae_um"], 6),
                }
            ),
            flush=True,
        )

    rows = []
    for record in pool["records"]:
        digest = record["parameter_sha256"]
        diagnostics = diagnostics_by_hash[digest]["diagnostics"]
        parameters = np.asarray(parameters_by_hash[digest], dtype=np.float64)
        coefficients = np.asarray(phase_coefficients_b7(jnp.asarray(parameters)))
        row = {
            **record,
            "objective": objective_from_diagnostics(diagnostics, record["profile"]),
            **diagnostics,
            "start_residual_mae_um": run_by_name[record["source_run"]][
                "start_validation"
            ]["residual_mae_um"],
            "seven_finite_phase_parameters": bool(
                parameters.shape == (7,) and np.all(np.isfinite(parameters))
            ),
            "physical_coefficient_norm_radians": float(np.linalg.norm(coefficients)),
            "minimum_support_energy_fraction": supports[digest],
        }
        row["eligible"] = is_soft_eligible(row)
        rows.append(row)

    selection = select_checkpoints(rows)
    selected_names = set(selection["selected_names"])
    by_name = {row["name"]: row for row in rows}
    selected = [
        {**by_name[name], "parameters": parameters_by_name[name]}
        for name in selection["selected_names"]
    ]
    trajectory_summaries = []
    for run in source["exact_runs"]:
        run_rows = sorted(
            [row for row in rows if row["source_run"] == run["name"]],
            key=lambda row: row["step"],
        )
        first_above = next(
            (
                row["step"]
                for row in run_rows
                if row["first_segmentation_loss"] > FIRST_SEGMENTATION_MAXIMUM
            ),
            None,
        )
        trajectory_summaries.append(
            {
                "source_run": run["name"],
                "first_step_above_first_loss_ceiling": first_above,
                "eligible_intervals": eligible_intervals(run_rows),
                "eligible_step_count": sum(row["eligible"] for row in run_rows),
            }
        )

    reproduction_rows = []
    reproduction_keys = (
        "objective",
        "first_segmentation_loss",
        "final_segmentation_loss",
        "focus_mse",
        "residual_mae_um",
        "normalized_residual_depth_squared",
        "normalized_stage_action_squared",
        "first_photon_mean",
        "second_photon_mean",
    )
    for run in source["exact_runs"]:
        current = by_name[f"{run['name']}-step-30"]
        differences = {
            key: abs(float(current[key]) - float(run["validation"][key]))
            for key in reproduction_keys
        }
        reproduction_rows.append(
            {
                "source_run": run["name"],
                "absolute_differences": differences,
                "maximum_absolute_difference": max(differences.values()),
            }
        )
    maximum_reproduction_difference = max(
        row["maximum_absolute_difference"] for row in reproduction_rows
    )
    if not math.isfinite(maximum_reproduction_difference) or (
        maximum_reproduction_difference > ENDPOINT_REPRODUCTION_TOLERANCE
    ):
        raise SystemExit("V2.4 pipeline failed to reproduce frozen v2.3 endpoints")

    duplicate_groups = [
        {
            "parameter_sha256": digest,
            "names": [row["name"] for row in records],
        }
        for digest, records in records_by_hash.items()
        if len(records) > 1
    ]
    report = {
        "status": "complete_soft_checkpoint_audit",
        "test_accessed": False,
        "training_rerun": False,
        "hard_labels_accessed": False,
        "source_artifact_sha256": pool["source_artifact_sha256"],
        "first_segmentation_maximum": FIRST_SEGMENTATION_MAXIMUM,
        "checkpoint_record_count": len(rows),
        "unique_parameter_hash_count": len(unique_hashes),
        "served_unique_evaluation_count": len(unique_hashes),
        "served_elapsed_seconds": float(
            sum(row["elapsed_seconds"] for row in diagnostics_by_hash.values())
        ),
        "duplicate_groups": duplicate_groups,
        "endpoint_reproduction_tolerance": ENDPOINT_REPRODUCTION_TOLERANCE,
        "maximum_endpoint_reproduction_absolute_difference": (
            maximum_reproduction_difference
        ),
        "endpoint_reproduction": reproduction_rows,
        "rows": rows,
        "trajectory_summaries": trajectory_summaries,
        "selection": selection,
        "selected": selected,
        "next_gate": (
            "freeze_selected_before_stopped_and_hard_validation"
            if selected
            else "freeze_negative_v2_4_and_activate_v2_5"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    render_trajectory_figures(rows, source, selected_names)
    render_tradeoff_figure(
        rows,
        selected_names,
        "first_segmentation_loss",
        "First-frame segmentation loss",
        "first-loss-vs-residual-mae.png",
    )
    render_tradeoff_figure(
        rows,
        selected_names,
        "final_segmentation_loss",
        "Final-frame segmentation loss",
        "final-loss-vs-residual-mae.png",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "checkpoint_records": len(rows),
                "unique_evaluations": len(unique_hashes),
                "eligible": len(selection["eligible_names"]),
                "nondominated": len(selection["nondominated_names"]),
                "selected": selection["selected_names"],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
