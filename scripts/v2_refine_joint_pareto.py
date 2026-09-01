"""Refine lower-focus-weight Pareto candidates after the first hard validation gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from tessscope.optics.model import phase_coefficients
from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    collect_patches,
    joint_value,
    joint_value_and_gradient,
    materialize_joint_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "optimization" / "summary.json"
)
SEGMENTATION_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "optimization"
    / "pareto-refinement.json"
)
JOINT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "optimization"
    / "pareto-refinement-from-joint.json"
)
FOCUS_WEIGHTS = (0.03, 0.05, 0.07)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--initialization",
        choices=("segmentation", "joint"),
        default="segmentation",
    )
    return parser.parse_args()


def validation_average(services, parameters, batches, calibration, weights):
    rows = []
    for batch in batches:
        value, branches = joint_value(
            *services, parameters, batch, calibration, weights
        )
        rows.append({"objective": value, **branches})
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    output = SEGMENTATION_OUTPUT if args.initialization == "segmentation" else JOINT_OUTPUT
    if output.exists():
        raise SystemExit(f"Pareto refinement already exists: {output}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    optimization = json.loads(OPTIMIZATION.read_text())
    if args.initialization == "segmentation":
        start_design = optimization["designs"]["segmentation_only"]
    else:
        selected_name = optimization["designs"]["selected_exact_joint"]
        start_design = next(
            row
            for row in optimization["designs"]["exact_joint_starts"]
            if row["name"] == selected_name
        )
    start_parameters = np.asarray(start_design["final_parameters"], dtype=np.float32)
    training_patches = collect_patches("training", 36, seed=53)
    validation_patches = collect_patches("validation", 12, seed=53)
    training_batches = [
        materialize_joint_batch(training_patches[index : index + 3], noise_seed=1000 + index)
        for index in range(0, 36, 3)
    ]
    validation_batches = [
        materialize_joint_batch(validation_patches[index : index + 3], noise_seed=None)
        for index in range(0, 12, 3)
    ]
    candidates = []
    for focus_weight in FOCUS_WEIGHTS:
        weights = JointObjectiveWeights(focus=focus_weight)
        parameters = start_parameters.copy()
        first_moment = np.zeros(6, dtype=np.float32)
        second_moment = np.zeros(6, dtype=np.float32)
        trace = []
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
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
            parameters -= 0.01 * (first_moment / (1 - 0.9**step)) / (
                np.sqrt(second_moment / (1 - 0.999**step)) + 1e-8
            )
            trace.append(
                {
                    "step": step,
                    "objective": value,
                    "gradient_norm": float(np.linalg.norm(gradient)),
                    **branches,
                }
            )
            print(
                json.dumps(
                    {
                        "focus_weight": focus_weight,
                        "step": step,
                        "objective": round(value, 6),
                    }
                ),
                flush=True,
            )
        candidates.append(
            {
                "name": f"joint-refine-focus-{focus_weight:.2f}",
                "focus_weight": focus_weight,
                "steps": args.steps,
                "initialization": f"promoted {args.initialization} parameters",
                "elapsed_seconds": time.perf_counter() - started,
                "final_parameters": parameters.tolist(),
                "final_phase_coefficients": np.asarray(
                    phase_coefficients(jnp.asarray(parameters))
                ).tolist(),
                "validation": validation_average(
                    services,
                    parameters,
                    validation_batches,
                    calibration,
                    weights,
                ),
                "trace": trace,
            }
        )
    report = {
        "status": "complete_training_validation_only_after_first_hard_gate",
        "reason": (
            "First exact joint passed clear/Pareto/stage gates but exceeded the 0.01 "
            "hard-PQ drop allowed from segmentation-only."
        ),
        "test_accessed": False,
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "candidate_count": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
