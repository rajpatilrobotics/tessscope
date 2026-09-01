"""Optimize B7/B11 pupils with segmentation-primary projected gradients."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    collect_patches,
    joint_value_and_gradient,
    materialize_joint_batch,
)
from tessscope.v2_1.optimization.constrained import is_promotion_eligible
from tessscope.v2_1.optimization.projected import segmentation_primary_gradient
from tessscope.v2_1.optimization.served import (
    averaged_forward,
    b7_coefficient_list,
    b11_coefficient_list,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEPARATE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-separate-baselines.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-projected-gradient-candidates.json"
)
DEFAULT_BALANCES = (0.25, 0.5, 1.0)
CHECKPOINT_INTERVAL = 5
SEGMENTATION_MARGIN = 0.008
SLACK_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--basis-size", type=int, choices=(7, 11), default=7)
    parser.add_argument("--separate", type=Path, default=DEFAULT_SEPARATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--balances", type=float, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(
            f"B{args.basis_size} projected-gradient artifact already exists: "
            f"{args.output}"
        )
    if args.steps <= 0 or args.learning_rate <= 0:
        raise SystemExit("Step count and learning rate must be positive")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    separate = json.loads(args.separate.read_text())
    balances = tuple(args.balances or DEFAULT_BALANCES)
    if any(balance < 0 for balance in balances):
        raise SystemExit("Projection balances must be nonnegative")
    basis_label = f"b{args.basis_size}"
    coefficient_list = (
        b7_coefficient_list
        if args.basis_size == 7
        else b11_coefficient_list
    )
    start_parameters = np.asarray(
        separate["segmentation_only"]["final_parameters"], dtype=np.float32
    )
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
    training_anchor = averaged_forward(
        services,
        start_parameters,
        training_batches,
        calibration,
    )
    validation_anchor = separate["segmentation_only"]["validation"]
    training_segmentation_limit = (
        training_anchor["segmentation_loss"] + SEGMENTATION_MARGIN
    )
    validation_segmentation_limit = (
        validation_anchor["segmentation_loss"] + SEGMENTATION_MARGIN
    )
    candidates = []
    for balance in balances:
        parameters = start_parameters.copy()
        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        trace = []
        checkpoints = []
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            batch = training_batches[(step - 1) % len(training_batches)]
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
            projected = segmentation_primary_gradient(
                segmentation_gradient,
                focus_gradient,
                balance=balance,
            )
            gradient = projected.combined.astype(np.float32)
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            schedule = args.learning_rate * (
                0.3 if step > 2 * args.steps / 3 else 1.0
            )
            parameters -= schedule * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
            trace.append(
                {
                    "step": step,
                    "learning_rate": schedule,
                    "raw_branch_gradient_dot": projected.raw_dot,
                    "focus_projection_scale": projected.projection_scale,
                    "combined_gradient_norm": float(np.linalg.norm(gradient)),
                    "batch_segmentation_loss": segmentation[
                        "segmentation_loss"
                    ],
                    "batch_focus_mse": focus["focus_mse"],
                    "parameters": parameters.tolist(),
                    "phase_coefficients": coefficient_list(parameters),
                }
            )
            if step % CHECKPOINT_INTERVAL == 0 or step == args.steps:
                validation = averaged_forward(
                    services,
                    parameters,
                    validation_batches,
                    calibration,
                )
                checkpoint = {
                    "name": (
                        f"{basis_label}-projected-{balance:.2f}-step-{step}"
                    ),
                    "balance": balance,
                    "step": step,
                    "parameters": parameters.tolist(),
                    "phase_coefficients": coefficient_list(parameters),
                    "validation": validation,
                    "validation_segmentation_eligible": bool(
                        validation["segmentation_loss"]
                        <= validation_segmentation_limit
                    ),
                }
                checkpoints.append(checkpoint)
                print(
                    json.dumps(
                        {
                            "balance": balance,
                            "step": step,
                            "raw_dot": round(projected.raw_dot, 6),
                            "validation_segmentation_loss": round(
                                validation["segmentation_loss"], 6
                            ),
                            "validation_focus_mse": round(
                                validation["focus_mse"], 6
                            ),
                        }
                    ),
                    flush=True,
                )
        validation_contenders = sorted(
            [
                row
                for row in checkpoints
                if row["validation_segmentation_eligible"]
            ],
            key=lambda row: (
                row["validation"]["focus_mse"],
                row["validation"]["segmentation_loss"],
            ),
        )[:2]
        for contender in validation_contenders:
            training = averaged_forward(
                services,
                np.asarray(contender["parameters"], dtype=np.float32),
                training_batches,
                calibration,
            )
            training["constraint_slack"] = (
                training_segmentation_limit - training["segmentation_loss"]
            )
            contender["training"] = training
            contender["eligible"] = is_promotion_eligible(
                training["constraint_slack"],
                contender["validation"]["segmentation_loss"],
                validation_segmentation_limit,
                slack_tolerance=SLACK_TOLERANCE,
            )
        eligible = [row for row in validation_contenders if row["eligible"]]
        selected = (
            min(
                eligible,
                key=lambda row: (
                    row["validation"]["focus_mse"],
                    row["validation"]["segmentation_loss"],
                ),
            )
            if eligible
            else None
        )
        candidates.append(
            {
                "name": f"{basis_label}-projected-balance-{balance:.2f}",
                "balance": balance,
                "steps": args.steps,
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoints": checkpoints,
                "full_training_contenders": validation_contenders,
                "selected_checkpoint": (
                    selected["name"] if selected is not None else None
                ),
                "trace": trace,
            }
        )
    selected_rows = [
        next(
            row
            for row in candidate["full_training_contenders"]
            if row["name"] == candidate["selected_checkpoint"]
        )
        for candidate in candidates
        if candidate["selected_checkpoint"] is not None
    ]
    selected_rows.sort(
        key=lambda row: (
            row["validation"]["focus_mse"],
            row["validation"]["segmentation_loss"],
        )
    )
    report = {
        "status": (
            f"complete_training_validation_only_{basis_label}_projected_gradient"
        ),
        "test_accessed": False,
        "method": (
            "remove only the exact normalized-focus gradient component opposing the "
            "exact segmentation gradient, then norm-balance and update with matched Adam"
        ),
        "budget": {
            "basis_size": args.basis_size,
            "balances": list(balances),
            "steps_per_balance": args.steps,
            "initial_learning_rate": args.learning_rate,
            "training_wells": len(training_patches),
            "validation_wells": len(validation_patches),
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        "constraints": {
            "segmentation_margin": SEGMENTATION_MARGIN,
            "training_segmentation_anchor": training_anchor[
                "segmentation_loss"
            ],
            "training_segmentation_limit": training_segmentation_limit,
            "validation_segmentation_anchor": validation_anchor[
                "segmentation_loss"
            ],
            "validation_segmentation_limit": validation_segmentation_limit,
            "minimum_training_constraint_slack": -SLACK_TOLERANCE,
        },
        "candidates": candidates,
        "selected_for_expanded_hard_validation": [
            row["name"] for row in selected_rows[:2]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_for_expanded_hard_validation": report[
                    "selected_for_expanded_hard_validation"
                ],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not selected_rows:
        raise SystemExit(
            f"No projected B{args.basis_size} checkpoint passed training and "
            "validation constraints"
        )


if __name__ == "__main__":
    main()
