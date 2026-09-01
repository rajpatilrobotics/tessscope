"""Optimize exact joint pupils with segmentation-primary projected branch gradients."""

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
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "optimization"
    / "projected-gradient-candidates.json"
)
BALANCES = (0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--steps", type=int, default=30)
    return parser.parse_args()


def validation_average(services, parameters, batches, calibration):
    rows = []
    weights = JointObjectiveWeights(focus=0.1)
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
    if OUTPUT.exists():
        raise SystemExit(f"Projected-gradient artifact already exists: {OUTPUT}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    optimization = json.loads(OPTIMIZATION.read_text())
    start_parameters = np.asarray(
        optimization["designs"]["segmentation_only"]["final_parameters"],
        dtype=np.float32,
    )
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
    for balance in BALANCES:
        parameters = start_parameters.copy()
        first_moment = np.zeros(6, dtype=np.float32)
        second_moment = np.zeros(6, dtype=np.float32)
        trace = []
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            batch = training_batches[(step - 1) % len(training_batches)]
            _, segmentation_gradient, segmentation_branches = joint_value_and_gradient(
                *services,
                parameters,
                batch,
                calibration,
                JointObjectiveWeights(segmentation=1.0, focus=0.0),
            )
            _, focus_gradient, focus_branches = joint_value_and_gradient(
                *services,
                parameters,
                batch,
                calibration,
                JointObjectiveWeights(segmentation=0.0, focus=1.0),
            )
            raw_dot = float(np.dot(segmentation_gradient, focus_gradient))
            projected_focus = focus_gradient.copy()
            if raw_dot < 0:
                projected_focus -= (
                    raw_dot
                    / max(float(np.dot(segmentation_gradient, segmentation_gradient)), 1e-12)
                    * segmentation_gradient
                )
            scale = balance * np.linalg.norm(segmentation_gradient) / max(
                np.linalg.norm(projected_focus), 1e-12
            )
            gradient = segmentation_gradient + scale * projected_focus
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
            parameters -= 0.01 * (first_moment / (1 - 0.9**step)) / (
                np.sqrt(second_moment / (1 - 0.999**step)) + 1e-8
            )
            trace.append(
                {
                    "step": step,
                    "raw_branch_gradient_dot": raw_dot,
                    "focus_projection_scale": float(scale),
                    "combined_gradient_norm": float(np.linalg.norm(gradient)),
                    "segmentation_loss": segmentation_branches["segmentation_loss"],
                    "focus_mse": focus_branches["focus_mse"],
                }
            )
            print(
                json.dumps(
                    {
                        "gradient_balance": balance,
                        "step": step,
                        "raw_dot": round(raw_dot, 6),
                    }
                ),
                flush=True,
            )
        candidates.append(
            {
                "name": f"projected-exact-joint-{balance:.2f}",
                "gradient_balance": balance,
                "steps": args.steps,
                "initialization": "promoted segmentation-only parameters",
                "elapsed_seconds": time.perf_counter() - started,
                "final_parameters": parameters.tolist(),
                "final_phase_coefficients": np.asarray(
                    phase_coefficients(jnp.asarray(parameters))
                ).tolist(),
                "validation": validation_average(
                    services, parameters, validation_batches, calibration
                ),
                "trace": trace,
            }
        )
    report = {
        "status": "complete_training_validation_only",
        "method": (
            "exact branch VJPs; remove the focus-gradient component opposing the "
            "segmentation gradient, then norm-balance the remaining focus direction"
        ),
        "test_accessed": False,
        "candidates": candidates,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "candidate_count": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
