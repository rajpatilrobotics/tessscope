"""Optimize matched B7 segmentation-only and focus-only baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
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
from tessscope.v2_1.optics.model import (
    phase_coefficients_b7,
    unconstrained_from_coefficients_b7,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_OPTIMIZATION = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "optimization" / "summary.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-separate-baselines.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


def coefficient_list(parameters: np.ndarray) -> list[float]:
    return np.asarray(phase_coefficients_b7(jnp.asarray(parameters))).tolist()


def extend_b6(coefficients: list[float]) -> np.ndarray:
    return unconstrained_from_coefficients_b7(
        np.asarray([*coefficients, 0.0], dtype=np.float64)
    )


def average_validation(services, parameters, batches, calibration, weights) -> dict:
    rows = []
    for batch in batches:
        value, branches = joint_value(
            *services,
            parameters,
            batch,
            calibration,
            weights,
        )
        rows.append({"objective": value, **branches})
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def optimize(
    name,
    services,
    start_parameters,
    training_batches,
    validation_batches,
    calibration,
    weights,
    *,
    steps,
    learning_rate,
) -> dict:
    parameters = np.asarray(start_parameters, dtype=np.float32).copy()
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    trace = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = training_batches[(step - 1) % len(training_batches)]
        value, gradient, branches = joint_value_and_gradient(
            *services,
            parameters,
            batch,
            calibration,
            weights,
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        schedule = learning_rate * (0.3 if step > 2 * steps / 3 else 1.0)
        parameters -= schedule * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        trace.append(
            {
                "step": step,
                "learning_rate": schedule,
                "training_objective": value,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "parameters": parameters.tolist(),
                "phase_coefficients": coefficient_list(parameters),
                **branches,
            }
        )
        print(
            json.dumps(
                {
                    "design": name,
                    "step": step,
                    "objective": round(value, 6),
                    "gradient_norm": round(float(np.linalg.norm(gradient)), 6),
                }
            ),
            flush=True,
        )
    return {
        "name": name,
        "optimizer": "Adam exact served gradient with final-third 0.3x decay",
        "weights": {
            "segmentation": weights.segmentation,
            "focus": weights.focus,
        },
        "steps": steps,
        "initial_learning_rate": learning_rate,
        "elapsed_seconds": time.perf_counter() - started,
        "start_parameters": np.asarray(start_parameters).tolist(),
        "start_phase_coefficients": coefficient_list(start_parameters),
        "final_parameters": parameters.tolist(),
        "final_phase_coefficients": coefficient_list(parameters),
        "validation": average_validation(
            services,
            parameters,
            validation_batches,
            calibration,
            weights,
        ),
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"B7 separate-baseline artifact already exists: {OUTPUT}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    training_patches = collect_patches("training", 36, seed=53)
    validation_patches = collect_patches("validation", 12, seed=53)
    training_batches = [
        materialize_joint_batch(
            training_patches[index : index + 3],
            noise_seed=2100 + index,
        )
        for index in range(0, len(training_patches), 3)
    ]
    validation_batches = [
        materialize_joint_batch(validation_patches[index : index + 3], noise_seed=None)
        for index in range(0, len(validation_patches), 3)
    ]
    v2 = json.loads(V2_OPTIMIZATION.read_text())["designs"]
    segmentation = optimize(
        "b7-segmentation-only",
        services,
        extend_b6(v2["segmentation_only"]["final_phase_coefficients"]),
        training_batches,
        validation_batches,
        calibration,
        JointObjectiveWeights(segmentation=1.0, focus=0.0),
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    focus = optimize(
        "b7-focus-only",
        services,
        extend_b6(v2["focus_only"]["final_phase_coefficients"]),
        training_batches,
        validation_batches,
        calibration,
        JointObjectiveWeights(segmentation=0.0, focus=1.0),
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    superposed_coefficients = np.asarray(
        segmentation["final_phase_coefficients"]
    ) + np.asarray(focus["final_phase_coefficients"])
    superposed_norm = float(np.linalg.norm(superposed_coefficients))
    if superposed_norm >= 2.5:
        superposed_coefficients *= 2.5 * 0.999 / superposed_norm
    superposed_parameters = unconstrained_from_coefficients_b7(superposed_coefficients)
    superposed = {
        "name": "b7-naive-superposition",
        "source_designs": [segmentation["name"], focus["name"]],
        "final_parameters": superposed_parameters.tolist(),
        "final_phase_coefficients": superposed_coefficients.tolist(),
        "validation": average_validation(
            services,
            superposed_parameters,
            validation_batches,
            calibration,
            JointObjectiveWeights(),
        ),
    }
    report = {
        "status": "complete_training_validation_only",
        "test_accessed": False,
        "budget": {
            "steps_per_design": args.steps,
            "initial_learning_rate": args.learning_rate,
            "training_batches": len(training_batches),
            "validation_batches": len(validation_batches),
        },
        "segmentation_only": segmentation,
        "focus_only": focus,
        "naive_superposition": superposed,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "segmentation_validation": segmentation["validation"],
                "focus_validation": focus["validation"],
                "superposition_validation": superposed["validation"],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
