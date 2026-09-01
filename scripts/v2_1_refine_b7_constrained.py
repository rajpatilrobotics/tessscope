"""Refine the eligible B7 constrained start on the full training-well budget."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    collect_patches,
    materialize_joint_batch,
)
from tessscope.v2_1.optimization.constrained import (
    BranchEvaluation,
    CachedEpsilonConstraint,
    is_promotion_eligible,
    solve_slsqp,
)
from tessscope.v2_1.optimization.served import (
    averaged_branches,
    averaged_forward,
    b7_coefficient_list,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_ROOT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2_1" / "optimization"
)
SCREEN = OPTIMIZATION_ROOT / "b7-constrained-screen.json"
SEPARATE = OPTIMIZATION_ROOT / "b7-separate-baselines.json"
DEFAULT_OUTPUT = OPTIMIZATION_ROOT / "b7-constrained-full-training.json"
DEFAULT_CONSTRAINT_MARGIN = 0.008
SLACK_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--maximum-iterations", type=int, default=10)
    parser.add_argument("--trust-radius", type=float, default=0.35)
    parser.add_argument(
        "--constraint-margin", type=float, default=DEFAULT_CONSTRAINT_MARGIN
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(
            f"Full-training constrained artifact already exists: {args.output}"
        )
    if args.constraint_margin <= 0:
        raise SystemExit("Constraint margin must be positive")
    screen = load_json(SCREEN)
    separate = load_json(SEPARATE)
    selected = screen["selected_for_expanded_training"]
    if len(selected) != 1:
        raise SystemExit(f"Expected one screened B7 candidate, found {selected}")
    candidate = next(
        row for row in screen["candidates"] if row["name"] == selected[0]
    )
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
    segmentation_parameters = np.asarray(
        separate["segmentation_only"]["final_parameters"], dtype=np.float32
    )
    training_anchor = averaged_forward(
        services,
        segmentation_parameters,
        training_batches,
        calibration,
    )
    validation_anchor = separate["segmentation_only"]["validation"]
    segmentation_limit = (
        training_anchor["segmentation_loss"] + args.constraint_margin
    )
    validation_segmentation_limit = (
        validation_anchor["segmentation_loss"] + args.constraint_margin
    )
    evaluation_count = 0

    def evaluator(parameters: np.ndarray) -> BranchEvaluation:
        nonlocal evaluation_count
        evaluation_count += 1
        evaluation = averaged_branches(
            services,
            parameters,
            training_batches,
            calibration,
        )
        print(
            json.dumps(
                {
                    "evaluation": evaluation_count,
                    "segmentation_loss": round(
                        evaluation.segmentation_loss, 6
                    ),
                    "focus_mse": round(evaluation.focus_mse, 6),
                    "constraint_slack": round(
                        segmentation_limit - evaluation.segmentation_loss,
                        6,
                    ),
                }
            ),
            flush=True,
        )
        return evaluation

    problem = CachedEpsilonConstraint(evaluator, segmentation_limit)
    start_parameters = np.asarray(candidate["final_parameters"], dtype=np.float32)
    started = time.perf_counter()
    result, trace = solve_slsqp(
        problem,
        start_parameters,
        trust_radius=args.trust_radius,
        maximum_iterations=args.maximum_iterations,
    )
    final_parameters = np.asarray(result.x, dtype=np.float32)
    final_training = problem.evaluate(final_parameters)
    validation = averaged_forward(
        services,
        final_parameters,
        validation_batches,
        calibration,
    )
    constraint_slack = segmentation_limit - final_training.segmentation_loss
    eligible = is_promotion_eligible(
        constraint_slack,
        validation["segmentation_loss"],
        validation_segmentation_limit,
        slack_tolerance=SLACK_TOLERANCE,
    )
    design_name = "b7-exact-constrained-full-training"
    report = {
        "status": "complete_full_training_validation_only",
        "test_accessed": False,
        "source_candidate": candidate["name"],
        "design": {
            "name": design_name,
            "optimizer": "SciPy SLSQP with exact served three-Tesseract Jacobians",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "jacobian_evaluations": int(result.njev),
            "exact_branch_evaluations": problem.cache_misses,
            "elapsed_seconds": time.perf_counter() - started,
            "start_parameters": start_parameters.tolist(),
            "final_parameters": final_parameters.tolist(),
            "final_phase_coefficients": b7_coefficient_list(final_parameters),
            "training": {
                "segmentation_loss": final_training.segmentation_loss,
                "focus_mse": final_training.focus_mse,
                "constraint_slack": constraint_slack,
            },
            "validation": validation,
            "eligible_for_hard_validation": eligible,
            "trace": trace,
        },
        "budget": {
            "training_wells": len(training_patches),
            "training_batches": len(training_batches),
            "validation_wells": len(validation_patches),
            "validation_batches": len(validation_batches),
            "maximum_iterations": args.maximum_iterations,
            "trust_radius": args.trust_radius,
        },
        "constraints": {
            "margin": args.constraint_margin,
            "training_segmentation_anchor": training_anchor[
                "segmentation_loss"
            ],
            "training_segmentation_limit": segmentation_limit,
            "validation_segmentation_anchor": validation_anchor[
                "segmentation_loss"
            ],
            "validation_segmentation_limit": validation_segmentation_limit,
            "minimum_training_constraint_slack": -SLACK_TOLERANCE,
        },
        "selected_for_hard_validation": design_name if eligible else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_for_hard_validation": report[
                    "selected_for_hard_validation"
                ],
                "training": report["design"]["training"],
                "validation": validation,
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not eligible:
        raise SystemExit(
            "Full-training B7 candidate is ineligible; hard validation remains blocked"
        )


if __name__ == "__main__":
    main()
