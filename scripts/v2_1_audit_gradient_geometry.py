"""Measure exact branch-gradient geometry across the frozen v2 Pareto path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tesseract_core import Tesseract

from tessscope.optics.model import phase_coefficients
from tessscope.v2.optics.model import unconstrained_from_coefficients
from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    collect_patches,
    joint_value,
    joint_value_and_gradient,
    materialize_joint_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "v2"
OUTPUT = PROJECT_ROOT / "artifacts" / "runs" / "v2_1" / "audit" / "gradient-geometry.json"
FIGURE = PROJECT_ROOT / "outputs" / "v2_1" / "gradient-geometry.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def named_parameters() -> list[tuple[str, np.ndarray]]:
    optimization = load_json(V2_ROOT / "optimization" / "summary.json")["designs"]
    exact = next(
        row
        for row in optimization["exact_joint_starts"]
        if row["name"] == optimization["selected_exact_joint"]
    )
    near = next(
        row
        for row in load_json(V2_ROOT / "optimization" / "pareto-refinement-from-joint.json")[
            "candidates"
        ]
        if row["name"] == "joint-refine-focus-0.03"
    )
    return [
        (
            "segmentation_only",
            np.asarray(optimization["segmentation_only"]["final_parameters"], dtype=np.float32),
        ),
        ("joint_refine_0.03", np.asarray(near["final_parameters"], dtype=np.float32)),
        ("exact_joint", np.asarray(exact["final_parameters"], dtype=np.float32)),
        (
            "naive_superposition",
            np.asarray(
                optimization["naive_superposition"]["final_parameters"],
                dtype=np.float32,
            ),
        ),
        (
            "astigmatic",
            np.asarray(
                optimization["classical_astigmatic"]["final_parameters"],
                dtype=np.float32,
            ),
        ),
        (
            "focus_only",
            np.asarray(optimization["focus_only"]["final_parameters"], dtype=np.float32),
        ),
    ]


def coefficient_vector(parameters: np.ndarray) -> np.ndarray:
    return np.asarray(phase_coefficients(jnp.asarray(parameters)), dtype=np.float64)


def gradient_row(services, parameters, batch, calibration) -> dict:
    _, segmentation_gradient, segmentation = joint_value_and_gradient(
        *services,
        parameters,
        batch,
        calibration,
        JointObjectiveWeights(segmentation=1.0, focus=0.0),
    )
    _, focus_gradient, focus = joint_value_and_gradient(
        *services,
        parameters,
        batch,
        calibration,
        JointObjectiveWeights(segmentation=0.0, focus=1.0),
    )
    # The shared joint objective reports raw focus MSE but differentiates MSE/6².
    # Convert its returned gradient back to the raw-MSE scale used in this audit.
    focus_gradient = focus_gradient * JointObjectiveWeights().maximum_depth_um ** 2
    dot = float(np.dot(segmentation_gradient, focus_gradient))
    denominator = max(
        float(np.linalg.norm(segmentation_gradient) * np.linalg.norm(focus_gradient)),
        1e-12,
    )
    return {
        "parameters": np.asarray(parameters).tolist(),
        "phase_coefficients": coefficient_vector(parameters).tolist(),
        "segmentation_loss": segmentation["segmentation_loss"],
        "focus_mse": focus["focus_mse"],
        "segmentation_gradient": segmentation_gradient.tolist(),
        "focus_gradient": focus_gradient.tolist(),
        "segmentation_gradient_norm": float(np.linalg.norm(segmentation_gradient)),
        "focus_gradient_norm": float(np.linalg.norm(focus_gradient)),
        "branch_gradient_dot": dot,
        "branch_gradient_cosine": dot / denominator,
        "gradients_conflict": dot < 0.0,
    }


def path_rows(services, nodes, batch, calibration) -> list[dict]:
    output = []
    distance = 0.0
    for segment_index, ((start_name, start), (end_name, end)) in enumerate(
        zip(nodes[:-1], nodes[1:], strict=True)
    ):
        start_coefficients = coefficient_vector(start)
        end_coefficients = coefficient_vector(end)
        segment_length = float(np.linalg.norm(end_coefficients - start_coefficients))
        samples = (0.0, 0.5) if segment_index < len(nodes) - 2 else (0.0, 0.5, 1.0)
        for fraction in samples:
            coefficients = (1.0 - fraction) * start_coefficients + fraction * end_coefficients
            parameters = unconstrained_from_coefficients(coefficients)
            row = gradient_row(services, parameters, batch, calibration)
            tangent = end - start
            row.update(
                {
                    "segment": f"{start_name}->{end_name}",
                    "start": start_name,
                    "end": end_name,
                    "fraction": fraction,
                    "path_distance_radians": distance + fraction * segment_length,
                    "segmentation_directional_derivative": float(
                        np.dot(row["segmentation_gradient"], tangent)
                    ),
                    "focus_directional_derivative": float(np.dot(row["focus_gradient"], tangent)),
                }
            )
            output.append(row)
            print(
                json.dumps(
                    {
                        "segment": row["segment"],
                        "fraction": fraction,
                        "branch_cosine": round(row["branch_gradient_cosine"], 6),
                    }
                ),
                flush=True,
            )
        distance += segment_length
    return output


def local_sweeps(services, near_parameters, batch, calibration) -> dict:
    center = gradient_row(services, near_parameters, batch, calibration)
    segmentation_gradient = np.asarray(center["segmentation_gradient"], dtype=np.float64)
    focus_gradient = np.asarray(center["focus_gradient"], dtype=np.float64)

    def normalized(value: np.ndarray) -> np.ndarray:
        return value / max(float(np.linalg.norm(value)), 1e-12)

    projected_focus = focus_gradient.copy()
    dot = float(np.dot(segmentation_gradient, focus_gradient))
    if dot < 0:
        projected_focus -= (
            dot
            / max(float(np.dot(segmentation_gradient, segmentation_gradient)), 1e-12)
            * segmentation_gradient
        )
    directions = {
        "negative_segmentation_gradient": normalized(-segmentation_gradient),
        "negative_focus_gradient": normalized(-focus_gradient),
        "negative_equal_norm_sum": normalized(
            -normalized(segmentation_gradient) - normalized(focus_gradient)
        ),
        "negative_projected_focus": normalized(-projected_focus),
    }
    for index in range(len(near_parameters)):
        basis = np.zeros_like(near_parameters, dtype=np.float64)
        basis[index] = 1.0
        directions[f"parameter_basis_{index + 1}"] = basis

    rows = []
    steps = (0.001, 0.005, 0.02)
    for name, direction in directions.items():
        for step in steps:
            for sign in (-1.0, 1.0):
                displacement = sign * step * direction
                parameters = np.asarray(near_parameters + displacement, dtype=np.float32)
                _, segmentation = joint_value(
                    *services,
                    parameters,
                    batch,
                    calibration,
                    JointObjectiveWeights(segmentation=1.0, focus=0.0),
                )
                _, focus = joint_value(
                    *services,
                    parameters,
                    batch,
                    calibration,
                    JointObjectiveWeights(segmentation=0.0, focus=1.0),
                )
                rows.append(
                    {
                        "direction": name,
                        "signed_step": sign * step,
                        "predicted_segmentation_delta": float(
                            np.dot(segmentation_gradient, displacement)
                        ),
                        "actual_segmentation_delta": (
                            segmentation["segmentation_loss"] - center["segmentation_loss"]
                        ),
                        "predicted_focus_delta": float(np.dot(focus_gradient, displacement)),
                        "actual_focus_delta": focus["focus_mse"] - center["focus_mse"],
                        "phase_coefficients": coefficient_vector(parameters).tolist(),
                    }
                )
    return {"center": center, "steps": steps, "rows": rows}


def plot_geometry(rows: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.plot(
        [row["focus_mse"] for row in rows],
        [row["segmentation_loss"] for row in rows],
        color="0.45",
        linewidth=1.2,
    )
    conflict = np.asarray([row["gradients_conflict"] for row in rows])
    colors = np.where(conflict, "tab:red", "tab:blue")
    axis.scatter(
        [row["focus_mse"] for row in rows],
        [row["segmentation_loss"] for row in rows],
        c=colors,
        s=50,
    )
    for row in rows:
        if row["fraction"] in {0.0, 1.0}:
            label = row["start"] if row["fraction"] == 0.0 else row["end"]
            axis.annotate(
                label,
                (row["focus_mse"], row["segmentation_loss"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Differentiable focus MSE (µm²)")
    axis.set_ylabel("Differentiable InstanSeg task loss")
    axis.set_title("Exact branch-gradient geometry along the B6 design path")
    axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for target in (OUTPUT, FIGURE):
        if target.exists() and not args.force:
            raise SystemExit(f"Gradient audit output already exists: {target}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    patches = collect_patches("validation", 3, seed=53)
    batch = materialize_joint_batch(patches, noise_seed=None)
    nodes = named_parameters()
    rows = path_rows(services, nodes, batch, calibration)
    near_parameters = dict(nodes)["joint_refine_0.03"]
    sweeps = local_sweeps(services, near_parameters, batch, calibration)
    conflicts = [row for row in rows if row["gradients_conflict"]]
    report = {
        "status": "complete_validation_only_exact_gradient_audit",
        "test_accessed": False,
        "batch_patch_ids": batch.patch_ids,
        "path_order": [name for name, _ in nodes],
        "path_rows": rows,
        "local_sweeps": sweeps,
        "summary": {
            "path_sample_count": len(rows),
            "conflict_fraction": len(conflicts) / len(rows),
            "minimum_branch_gradient_cosine": min(row["branch_gradient_cosine"] for row in rows),
            "maximum_branch_gradient_cosine": max(row["branch_gradient_cosine"] for row in rows),
        },
        "figure": str(FIGURE.relative_to(PROJECT_ROOT)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    plot_geometry(rows, FIGURE)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "figure": str(FIGURE),
                **report["summary"],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
