"""Run exact-gradient B7 epsilon-constraint continuation from four starts."""

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
from tessscope.v2_1.optics.model import (
    unconstrained_from_coefficients_b7,
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
V2_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "v2" / "optimization"
B7_SEPARATE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-separate-baselines.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-constrained-screen.json"
)
CONSTRAINT_MARGINS = (0.004, 0.008)
TRAINING_SLACK_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--maximum-iterations", type=int, default=10)
    parser.add_argument("--trust-radius", type=float, default=0.5)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def project_inside_ball(value: np.ndarray) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(candidate))
    if norm >= 2.5:
        candidate *= 2.5 * 0.999 / norm
    return candidate


def start_registry(separate: dict) -> dict[str, np.ndarray]:
    v2 = load_json(V2_ROOT / "summary.json")["designs"]
    exact = next(
        row
        for row in v2["exact_joint_starts"]
        if row["name"] == v2["selected_exact_joint"]
    )
    near = next(
        row
        for row in load_json(V2_ROOT / "pareto-refinement-from-joint.json")[
            "candidates"
        ]
        if row["name"] == "joint-refine-focus-0.03"
    )
    astigmatic = np.asarray(
        v2["classical_astigmatic"]["final_phase_coefficients"], dtype=np.float64
    )
    segmentation_coefficients = np.asarray(
        separate["segmentation_only"]["final_phase_coefficients"],
        dtype=np.float64,
    )
    near_coefficients = np.asarray([*near["final_phase_coefficients"], 0.25])
    exact_coefficients = np.asarray([*exact["final_phase_coefficients"], 0.25])
    segmentation_astigmatic = segmentation_coefficients.copy()
    segmentation_astigmatic[:6] += 0.35 * astigmatic
    return {
        "b7_segmentation_only": np.asarray(
            separate["segmentation_only"]["final_parameters"], dtype=np.float32
        ),
        "v2_near_plus_spherical_0.25": unconstrained_from_coefficients_b7(
            project_inside_ball(near_coefficients)
        ),
        "v2_exact_plus_spherical_0.25": unconstrained_from_coefficients_b7(
            project_inside_ball(exact_coefficients)
        ),
        "b7_segmentation_plus_0.35_astigmatic": unconstrained_from_coefficients_b7(
            project_inside_ball(segmentation_astigmatic)
        ),
    }


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"B7 constrained screen already exists: {OUTPUT}")
    if not B7_SEPARATE.exists():
        raise SystemExit(f"Run matched B7 separate baselines first: {B7_SEPARATE}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    training_patches = collect_patches("training", 6, seed=71)
    validation_patches = collect_patches("validation", 12, seed=53)
    training_batches = [
        materialize_joint_batch(
            training_patches[index : index + 3],
            noise_seed=3200 + index,
        )
        for index in range(0, len(training_patches), 3)
    ]
    validation_batches = [
        materialize_joint_batch(validation_patches[index : index + 3], noise_seed=None)
        for index in range(0, len(validation_patches), 3)
    ]
    separate = load_json(B7_SEPARATE)
    segmentation_parameters = np.asarray(
        separate["segmentation_only"]["final_parameters"], dtype=np.float32
    )
    training_anchor = averaged_branches(
        services,
        segmentation_parameters,
        training_batches,
        calibration,
    )
    starts = start_registry(separate)
    candidates = []
    for start_name, start in starts.items():
        parameters = start.copy()
        for margin in CONSTRAINT_MARGINS:
            segmentation_limit = training_anchor.segmentation_loss + margin
            evaluation_count = 0

            def evaluator(
                candidate: np.ndarray,
                *,
                _start_name: str = start_name,
                _margin: float = margin,
            ) -> BranchEvaluation:
                nonlocal evaluation_count
                evaluation_count += 1
                result = averaged_branches(
                    services,
                    candidate,
                    training_batches,
                    calibration,
                )
                print(
                    json.dumps(
                        {
                            "start": _start_name,
                            "margin": _margin,
                            "evaluation": evaluation_count,
                            "segmentation_loss": round(
                                result.segmentation_loss, 6
                            ),
                            "focus_mse": round(result.focus_mse, 6),
                        }
                    ),
                    flush=True,
                )
                return result

            problem = CachedEpsilonConstraint(evaluator, segmentation_limit)
            started = time.perf_counter()
            result, trace = solve_slsqp(
                problem,
                parameters,
                trust_radius=args.trust_radius,
                maximum_iterations=args.maximum_iterations,
            )
            parameters = np.asarray(result.x, dtype=np.float32)
            final_training = problem.evaluate(parameters)
            candidates.append(
                {
                    "name": f"{start_name}-margin-{margin:.3f}",
                    "start_family": start_name,
                    "constraint_margin": margin,
                    "segmentation_limit": segmentation_limit,
                    "optimizer": "SciPy SLSQP with exact three-Tesseract Jacobians",
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "iterations": int(result.nit),
                    "function_evaluations": int(result.nfev),
                    "jacobian_evaluations": int(result.njev),
                    "exact_branch_evaluations": problem.cache_misses,
                    "elapsed_seconds": time.perf_counter() - started,
                    "final_parameters": parameters.tolist(),
                    "final_phase_coefficients": b7_coefficient_list(parameters),
                    "training": {
                        "segmentation_loss": final_training.segmentation_loss,
                        "focus_mse": final_training.focus_mse,
                        "constraint_slack": (
                            segmentation_limit - final_training.segmentation_loss
                        ),
                    },
                    "validation": averaged_forward(
                        services,
                        parameters,
                        validation_batches,
                        calibration,
                    ),
                    "trace": trace,
                }
            )
    validation_segmentation_anchor = separate["segmentation_only"]["validation"][
        "segmentation_loss"
    ]
    validation_segmentation_limit = validation_segmentation_anchor + 0.008
    eligible = [
        row
        for row in candidates
        if is_promotion_eligible(
            row["training"]["constraint_slack"],
            row["validation"]["segmentation_loss"],
            validation_segmentation_limit,
            slack_tolerance=TRAINING_SLACK_TOLERANCE,
        )
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            row["validation"]["focus_mse"],
            row["validation"]["segmentation_loss"],
        ),
    )
    report = {
        "status": "complete_training_validation_only_constrained_screen",
        "test_accessed": False,
        "method": (
            "focus MSE/36 objective subject to exact InstanSeg task-loss epsilon "
            "constraint, both using served three-Tesseract Jacobians"
        ),
        "budget": {
            "training_batches": len(training_batches),
            "validation_batches": len(validation_batches),
            "maximum_iterations_per_stage": args.maximum_iterations,
            "trust_radius_per_stage": args.trust_radius,
            "constraint_margins": list(CONSTRAINT_MARGINS),
        },
        "training_segmentation_anchor": training_anchor.segmentation_loss,
        "validation_segmentation_anchor": validation_segmentation_anchor,
        "validation_segmentation_limit": validation_segmentation_limit,
        "promotion_criteria": {
            "minimum_training_constraint_slack": -TRAINING_SLACK_TOLERANCE,
            "maximum_validation_segmentation_loss": validation_segmentation_limit,
        },
        "starts": {name: value.tolist() for name, value in starts.items()},
        "candidates": candidates,
        "eligible_ranking": [row["name"] for row in ranked],
        "selected_for_expanded_training": [row["name"] for row in ranked[:2]],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selected": report["selected_for_expanded_training"],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
