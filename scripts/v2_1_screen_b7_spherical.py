"""Soft validation screen for the added B7 primary-spherical coefficient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    collect_patches,
    joint_value,
    joint_value_and_gradient,
    materialize_joint_batch,
)
from tessscope.v2_1.optics.model import unconstrained_from_coefficients_b7

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "v2"
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "screening"
    / "b7-spherical-screen.json"
)
SPHERICAL_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def seed_coefficients() -> dict[str, np.ndarray]:
    optimization = load_json(V2_ROOT / "optimization" / "summary.json")["designs"]
    exact = next(
        row
        for row in optimization["exact_joint_starts"]
        if row["name"] == optimization["selected_exact_joint"]
    )
    near = next(
        row
        for row in load_json(
            V2_ROOT / "optimization" / "pareto-refinement-from-joint.json"
        )["candidates"]
        if row["name"] == "joint-refine-focus-0.03"
    )
    return {
        "segmentation_only": np.asarray(
            optimization["segmentation_only"]["final_phase_coefficients"]
        ),
        "joint_refine_0.03": np.asarray(near["final_phase_coefficients"]),
        "exact_joint": np.asarray(exact["final_phase_coefficients"]),
        "naive_superposition": np.asarray(
            optimization["naive_superposition"]["final_phase_coefficients"]
        ),
        "focus_only": np.asarray(optimization["focus_only"]["final_phase_coefficients"]),
    }


def validation_average(services, parameters, batches, calibration) -> dict:
    rows = []
    for batch in batches:
        _, branches = joint_value(
            *services,
            parameters,
            batch,
            calibration,
            JointObjectiveWeights(),
        )
        rows.append(branches)
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"B7 spherical screen already exists: {OUTPUT}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    patches = collect_patches("validation", 12, seed=53)
    batches = [
        materialize_joint_batch(patches[index : index + 3], noise_seed=None)
        for index in range(0, len(patches), 3)
    ]
    candidates = []
    gradients = []
    for seed_name, b6_coefficients in seed_coefficients().items():
        for spherical in SPHERICAL_GRID:
            coefficients = np.asarray([*b6_coefficients, spherical], dtype=np.float64)
            parameters = unconstrained_from_coefficients_b7(coefficients)
            validation = validation_average(
                services,
                parameters,
                batches,
                calibration,
            )
            candidates.append(
                {
                    "seed": seed_name,
                    "spherical_coefficient_radians": spherical,
                    "phase_coefficients": coefficients.tolist(),
                    "phase_rms_radians": float(np.linalg.norm(coefficients)),
                    "parameters": parameters.tolist(),
                    "validation": validation,
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed_name,
                        "spherical": spherical,
                        "segmentation_loss": validation["segmentation_loss"],
                        "focus_mse": validation["focus_mse"],
                    }
                ),
                flush=True,
            )
        center = unconstrained_from_coefficients_b7(
            np.asarray([*b6_coefficients, 0.0])
        )
        _, segmentation_gradient, segmentation = joint_value_and_gradient(
            *services,
            center,
            batches[0],
            calibration,
            JointObjectiveWeights(segmentation=1.0, focus=0.0),
        )
        _, focus_gradient, focus = joint_value_and_gradient(
            *services,
            center,
            batches[0],
            calibration,
            JointObjectiveWeights(segmentation=0.0, focus=1.0),
        )
        gradients.append(
            {
                "seed": seed_name,
                "batch_patch_ids": batches[0].patch_ids,
                "segmentation_loss": segmentation["segmentation_loss"],
                "focus_mse": focus["focus_mse"],
                "segmentation_parameter_gradient_spherical": float(
                    segmentation_gradient[6]
                ),
                "normalized_focus_parameter_gradient_spherical": float(
                    focus_gradient[6]
                ),
            }
        )
    improvements = []
    for seed_name in seed_coefficients():
        rows = [row for row in candidates if row["seed"] == seed_name]
        zero = next(
            row for row in rows if row["spherical_coefficient_radians"] == 0.0
        )
        for row in rows:
            if row is zero:
                continue
            improvements.append(
                {
                    "seed": seed_name,
                    "spherical_coefficient_radians": row[
                        "spherical_coefficient_radians"
                    ],
                    "segmentation_loss_delta": (
                        row["validation"]["segmentation_loss"]
                        - zero["validation"]["segmentation_loss"]
                    ),
                    "focus_mse_delta": (
                        row["validation"]["focus_mse"]
                        - zero["validation"]["focus_mse"]
                    ),
                    "strictly_dominates_zero": bool(
                        row["validation"]["segmentation_loss"]
                        < zero["validation"]["segmentation_loss"]
                        and row["validation"]["focus_mse"]
                        < zero["validation"]["focus_mse"]
                    ),
                }
            )
    report = {
        "status": "complete_validation_only_b7_spherical_screen",
        "test_accessed": False,
        "spherical_grid_radians": list(SPHERICAL_GRID),
        "validation_patch_ids": [patch.field_id for patch in patches],
        "candidates": candidates,
        "center_gradients": gradients,
        "relative_to_zero": improvements,
        "strict_dominance_count": sum(
            row["strictly_dominates_zero"] for row in improvements
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "strict_dominance_count": report["strict_dominance_count"],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
