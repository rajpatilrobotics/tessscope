"""Run the preregistered exact constrained B7/B11 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration, collect_patches
from tessscope.v2_3.closed_loop import (
    ClosedLoopWeights,
    closed_loop_value_and_gradient,
    materialize_closed_loop_batch,
)
from tessscope.v2_5.constrained import (
    CachedClosedLoopConstraints,
    ConstraintEvaluation,
    closed_loop_augmented_value_and_gradient,
    project_to_physical_radius,
    scaled_violations,
    solve_slsqp,
    update_duals,
)
from tessscope.v2_5.evaluation import (
    average_closed_loop_forward,
    parameter_sha256,
    physical_diagnostics,
)
from tessscope.v2_5.selection import (
    alternate_start_key,
    is_soft_eligible,
    select_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT_ROOT / "configs" / "v2_5" / "contract.yaml"
MANIFEST = PROJECT_ROOT / "configs" / "v2_5" / "source-manifest.json"
B7_SEPARATE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-separate-baselines.json"
)
B11_SEPARATE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b11-separate-baselines.json"
)
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "artifacts" / "runtime-runs" / "v2_5-constrained-aggregate"
)
RESIDUAL_BOUNDS = (0.075, 0.065, 0.055)
STEPS_PER_STAGE = 18
LEARNING_RATE = 0.008
FIRST_SCALE = 0.008
RESIDUAL_SCALE = 0.02
FINAL_WEIGHTS = ClosedLoopWeights(
    first_segmentation=0.0,
    final_segmentation=1.0,
    normalized_residual_depth_squared=0.0,
    normalized_stage_action_squared=0.0,
    fixed_second_exposure_cost=0.0,
)
FIRST_WEIGHTS = ClosedLoopWeights(
    first_segmentation=1.0,
    final_segmentation=0.0,
    normalized_residual_depth_squared=0.0,
    normalized_stage_action_squared=0.0,
    fixed_second_exposure_cost=0.0,
)
RESIDUAL_WEIGHTS = ClosedLoopWeights(
    first_segmentation=0.0,
    final_segmentation=0.0,
    normalized_residual_depth_squared=1.0,
    normalized_stage_action_squared=0.0,
    fixed_second_exposure_cost=0.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", choices=("b7", "b11"), required=True)
    parser.add_argument("--optics-url")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basis_settings(basis: str, manifest: dict) -> dict:
    if basis == "b7":
        separate = json.loads(B7_SEPARATE.read_text())
        start_names = {
            "b7_v2_4_balanced_step14",
            "b7_v2_2_piecewise_028",
        }
        return {
            "label": "B7",
            "size": 7,
            "starts": [
                row for row in manifest["starts"] if row["name"] in start_names
            ],
            "anchor_parameters": separate["segmentation_only"]["final_parameters"],
            "soft_maximum": manifest["anchors"][
                "b7_soft_first_segmentation_maximum"
            ],
            "default_url": "http://127.0.0.1:8407",
        }
    start_names = {
        "b11_lift_v2_4_balanced_step14",
        "b11_lift_v2_2_piecewise_028",
        "b11_segmentation_only",
        "b11_projected_2_step15",
        "b11_projected_3_step15",
    }
    return {
        "label": "B11",
        "size": 11,
        "starts": [row for row in manifest["starts"] if row["name"] in start_names],
        "anchor_parameters": json.loads(B11_SEPARATE.read_text())[
            "segmentation_only"
        ]["final_parameters"],
        "soft_maximum": manifest["anchors"][
            "b11_soft_first_segmentation_maximum"
        ],
        "default_url": "http://127.0.0.1:8408",
    }


def basis_soft_eligible(row: dict, *, soft_maximum: float) -> bool:
    return bool(
        row["validation"]["first_segmentation_loss"] <= soft_maximum
        and row["validation"]["normalized_residual_depth_squared"]
        <= row["residual_bound"] + 1e-4
        and row["validation"]["residual_mae_um"]
        < row["start_validation"]["residual_mae_um"]
        and row["physical"]["physical_coefficient_norm_radians"] < 2.5
        and row["physical"]["minimum_support_energy_fraction"] >= 0.995
    )


def penalty_for_step(step: int) -> float:
    return min(4.0 * (2.0 ** ((step - 1) // 6)), 32.0)


def optimize_stage(
    *,
    basis: str,
    basis_size: int,
    start_name: str,
    start_parameters: np.ndarray,
    residual_bound: float,
    first_limit: float,
    services,
    training_batches,
    validation_batches,
    calibration,
    original_start_validation: dict,
) -> dict:
    bound_label = f"{residual_bound:.3f}".replace(".", "p")
    name = f"v2_5-{basis}-auglag-{start_name}-residual-{bound_label}"
    checkpoint_path = CHECKPOINT_ROOT / basis / f"{name}.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("completed"):
            return checkpoint["result"]
        parameters = np.asarray(checkpoint["parameters"], dtype=np.float32)
        first_moment = np.asarray(checkpoint["first_moment"], dtype=np.float32)
        second_moment = np.asarray(checkpoint["second_moment"], dtype=np.float32)
        duals = np.asarray(checkpoint["duals"], dtype=np.float32)
        trace = checkpoint["trace"]
        first_step = int(checkpoint["next_step"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
        stage_start = np.asarray(checkpoint["stage_start"], dtype=np.float32)
        stage_start_validation = checkpoint["stage_start_validation"]
        projection_count = int(checkpoint["projection_count"])
    else:
        parameters = np.asarray(start_parameters, dtype=np.float32).copy()
        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        duals = np.zeros(2, dtype=np.float32)
        trace = []
        first_step = 1
        elapsed_before = 0.0
        stage_start = parameters.copy()
        stage_start_validation = average_closed_loop_forward(
            services, stage_start, validation_batches, calibration
        )
        projection_count = 0
    started = time.perf_counter()
    for step in range(first_step, STEPS_PER_STAGE + 1):
        penalty = penalty_for_step(step)
        value, gradient, diagnostics = closed_loop_augmented_value_and_gradient(
            *services,
            parameters,
            training_batches,
            calibration,
            first_limit=first_limit,
            residual_limit=residual_bound,
            duals=duals,
            penalty=penalty,
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        rate = LEARNING_RATE * (0.3 if step > 12 else 1.0)
        update = rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        update_norm = float(np.linalg.norm(update))
        if update_norm > 0.08:
            update *= 0.08 / update_norm
        parameters -= update
        parameters, projected = project_to_physical_radius(
            parameters, basis_size=basis_size
        )
        projection_count += int(projected)
        if step % 3 == 0:
            duals = update_duals(
                duals,
                np.asarray(
                    [
                        diagnostics["first_scaled_violation"],
                        diagnostics["residual_scaled_violation"],
                    ]
                ),
                penalty=penalty,
            )
        trace.append(
            {
                "step": step,
                "parameters": parameters.tolist(),
                "augmented_objective": value,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "applied_update_norm": float(np.linalg.norm(update)),
                "learning_rate": rate,
                "duals_after_update": duals.tolist(),
                "physical_projection_applied": projected,
                **diagnostics,
            }
        )
        elapsed = elapsed_before + time.perf_counter() - started
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "completed": False,
                    "parameters": parameters.tolist(),
                    "first_moment": first_moment.tolist(),
                    "second_moment": second_moment.tolist(),
                    "duals": duals.tolist(),
                    "trace": trace,
                    "next_step": step + 1,
                    "elapsed_seconds": elapsed,
                    "stage_start": stage_start.tolist(),
                    "stage_start_validation": stage_start_validation,
                    "projection_count": projection_count,
                }
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "name": name,
                    "step": step,
                    "of": STEPS_PER_STAGE,
                    "final_loss": round(diagnostics["final_segmentation_loss"], 6),
                    "first_violation": round(
                        diagnostics["first_scaled_violation"], 4
                    ),
                    "residual_violation": round(
                        diagnostics["residual_scaled_violation"], 4
                    ),
                }
            ),
            flush=True,
        )
    elapsed = elapsed_before + time.perf_counter() - started
    training = average_closed_loop_forward(
        services, parameters, training_batches, calibration
    )
    validation = average_closed_loop_forward(
        services, parameters, validation_batches, calibration
    )
    violations = scaled_violations(
        training["first_segmentation_loss"],
        training["normalized_residual_depth_squared"],
        first_limit=first_limit,
        residual_limit=residual_bound,
    )
    physical = physical_diagnostics(parameters, basis_size=basis_size)
    result = {
        "name": name,
        "basis": basis.upper(),
        "basis_size": basis_size,
        "method": "exact_gradient_augmented_lagrangian",
        "start_name": start_name,
        "residual_bound": residual_bound,
        "first_training_limit": first_limit,
        "steps": STEPS_PER_STAGE,
        "elapsed_seconds": elapsed,
        "stage_start_parameters": stage_start.tolist(),
        "final_parameters": parameters.tolist(),
        "parameter_sha256": parameter_sha256(parameters),
        "original_start_validation": original_start_validation,
        "stage_start_validation": stage_start_validation,
        "training": training,
        "validation": validation,
        "training_first_scaled_violation": float(violations[0]),
        "training_residual_scaled_violation": float(violations[1]),
        "training_final_segmentation_loss": training["final_segmentation_loss"],
        "physical": physical,
        "physical_projection_count": projection_count,
        "trace": trace,
    }
    checkpoint_path.write_text(
        json.dumps({"completed": True, "result": result}) + "\n"
    )
    return result


def averaged_constraint_evaluation(
    services,
    parameters: np.ndarray,
    batches,
    calibration,
) -> ConstraintEvaluation:
    values: dict[str, list] = {
        "final": [],
        "first": [],
        "residual": [],
        "final_gradient": [],
        "first_gradient": [],
        "residual_gradient": [],
    }
    for batch in batches:
        final, final_gradient, _ = closed_loop_value_and_gradient(
            *services, parameters, batch, calibration, FINAL_WEIGHTS
        )
        first, first_gradient, _ = closed_loop_value_and_gradient(
            *services, parameters, batch, calibration, FIRST_WEIGHTS
        )
        residual, residual_gradient, _ = closed_loop_value_and_gradient(
            *services, parameters, batch, calibration, RESIDUAL_WEIGHTS
        )
        values["final"].append(final)
        values["first"].append(first)
        values["residual"].append(residual)
        values["final_gradient"].append(final_gradient)
        values["first_gradient"].append(first_gradient)
        values["residual_gradient"].append(residual_gradient)
    return ConstraintEvaluation(
        final_segmentation_loss=float(np.mean(values["final"])),
        first_segmentation_loss=float(np.mean(values["first"])),
        normalized_residual_depth_squared=float(np.mean(values["residual"])),
        final_gradient=np.mean(values["final_gradient"], axis=0),
        first_gradient=np.mean(values["first_gradient"], axis=0),
        residual_gradient=np.mean(values["residual_gradient"], axis=0),
    )


def run_alternate(
    *,
    basis: str,
    basis_size: int,
    primary: list[dict],
    services,
    training_batches,
    validation_batches,
    calibration,
    first_limit: float,
) -> list[dict]:
    alternates = []
    for residual_bound in RESIDUAL_BOUNDS:
        source = min(
            [row for row in primary if row["residual_bound"] == residual_bound],
            key=alternate_start_key,
        )
        evaluation_count = 0

        def evaluator(
            parameters: np.ndarray,
            *,
            _source_name: str = source["name"],
            _residual_bound: float = residual_bound,
        ) -> ConstraintEvaluation:
            nonlocal evaluation_count
            evaluation_count += 1
            result = averaged_constraint_evaluation(
                services,
                np.asarray(parameters, dtype=np.float32),
                training_batches[:2],
                calibration,
            )
            print(
                json.dumps(
                    {
                        "alternate_source": _source_name,
                        "residual_bound": _residual_bound,
                        "evaluation": evaluation_count,
                        "final": round(result.final_segmentation_loss, 6),
                        "first": round(result.first_segmentation_loss, 6),
                        "residual": round(
                            result.normalized_residual_depth_squared, 6
                        ),
                    }
                ),
                flush=True,
            )
            return result

        problem = CachedClosedLoopConstraints(
            evaluator,
            first_limit=first_limit,
            residual_limit=residual_bound,
        )
        started = time.perf_counter()
        result, trace = solve_slsqp(
            problem,
            np.asarray(source["final_parameters"], dtype=np.float32),
        )
        parameters, projected = project_to_physical_radius(
            np.asarray(result.x, dtype=np.float32), basis_size=basis_size
        )
        training = average_closed_loop_forward(
            services, parameters, training_batches, calibration
        )
        validation = average_closed_loop_forward(
            services, parameters, validation_batches, calibration
        )
        violations = scaled_violations(
            training["first_segmentation_loss"],
            training["normalized_residual_depth_squared"],
            first_limit=first_limit,
            residual_limit=residual_bound,
        )
        alternates.append(
            {
                "name": (
                    f"v2_5-{basis}-slsqp-{source['start_name']}-"
                    f"residual-{residual_bound:.3f}".replace(".", "p")
                ),
                "basis": basis.upper(),
                "basis_size": basis_size,
                "method": "exact_jacobian_slsqp_alternate",
                "start_name": source["start_name"],
                "source_primary_name": source["name"],
                "residual_bound": residual_bound,
                "first_training_limit": first_limit,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "exact_vector_evaluations": problem.cache_misses,
                "elapsed_seconds": time.perf_counter() - started,
                "final_parameters": parameters.tolist(),
                "parameter_sha256": parameter_sha256(parameters),
                "original_start_validation": source["original_start_validation"],
                "training": training,
                "validation": validation,
                "training_first_scaled_violation": float(violations[0]),
                "training_residual_scaled_violation": float(violations[1]),
                "training_final_segmentation_loss": training[
                    "final_segmentation_loss"
                ],
                "physical": physical_diagnostics(
                    parameters, basis_size=basis_size
                ),
                "physical_projection_applied": projected,
                "trace": trace,
            }
        )
    return alternates


def selection_row(row: dict, *, first_maximum: float) -> dict:
    validation = row["validation"]
    physical = row["physical"]
    start = row["original_start_validation"]
    value = {
        "name": row["name"],
        "basis": row["basis"],
        "method": row["method"],
        "start_name": row["start_name"],
        "residual_bound": row["residual_bound"],
        "parameter_sha256": row["parameter_sha256"],
        "parameters": row["final_parameters"],
        "finite_parameters": bool(np.isfinite(row["final_parameters"]).all()),
        "first_segmentation_loss": validation["first_segmentation_loss"],
        "final_segmentation_loss": validation["final_segmentation_loss"],
        "residual_mae_um": validation["residual_mae_um"],
        "normalized_residual_depth_squared": validation[
            "normalized_residual_depth_squared"
        ],
        "start_residual_mae_um": start["residual_mae_um"],
        "physical_coefficient_norm_radians": physical[
            "physical_coefficient_norm_radians"
        ],
        "minimum_support_energy_fraction": physical[
            "minimum_support_energy_fraction"
        ],
    }
    value["eligible"] = is_soft_eligible(
        value, first_segmentation_maximum=first_maximum
    )
    return value


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    settings = basis_settings(args.basis, manifest)
    output = (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "v2_5"
        / "optimization"
        / f"{args.basis}-constrained.json"
    )
    if output.exists():
        raise SystemExit(f"V2.5 constrained artifact already exists: {output}")
    services = (
        Tesseract.from_url(args.optics_url or settings["default_url"], timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    training_patches = collect_patches("training", 12, seed=53)
    validation_patches = collect_patches("validation", 12, seed=53)
    training_batches = [
        materialize_closed_loop_batch(
            training_patches[index : index + 3], noise_seed=5200 + index
        )
        for index in range(0, len(training_patches), 3)
    ]
    validation_batches = [
        materialize_closed_loop_batch(
            validation_patches[index : index + 3], noise_seed=None
        )
        for index in range(0, len(validation_patches), 3)
    ]
    anchor_training = average_closed_loop_forward(
        services,
        np.asarray(settings["anchor_parameters"], dtype=np.float32),
        training_batches,
        calibration,
    )
    first_limit = anchor_training["first_segmentation_loss"] + 0.008
    primary = []
    for start in settings["starts"]:
        initial = np.asarray(start["parameters"], dtype=np.float32)
        original_start_validation = average_closed_loop_forward(
            services, initial, validation_batches, calibration
        )
        parameters = initial
        for residual_bound in RESIDUAL_BOUNDS:
            result = optimize_stage(
                basis=args.basis,
                basis_size=settings["size"],
                start_name=start["name"],
                start_parameters=parameters,
                residual_bound=residual_bound,
                first_limit=first_limit,
                services=services,
                training_batches=training_batches,
                validation_batches=validation_batches,
                calibration=calibration,
                original_start_validation=original_start_validation,
            )
            primary.append(result)
            parameters = np.asarray(result["final_parameters"], dtype=np.float32)
    primary_eligible = [
        row
        for row in primary
        if basis_soft_eligible(row, soft_maximum=settings["soft_maximum"])
    ]
    alternate = []
    if not primary_eligible:
        alternate = run_alternate(
            basis=args.basis,
            basis_size=settings["size"],
            primary=primary,
            services=services,
            training_batches=training_batches,
            validation_batches=validation_batches,
            calibration=calibration,
            first_limit=first_limit,
        )
    all_runs = [*primary, *alternate]
    selection_rows = []
    selection = None
    if args.basis == "b11":
        selection_rows = [
            selection_row(row, first_maximum=settings["soft_maximum"])
            for row in all_runs
        ]
        selection = select_candidates(selection_rows)
    report = {
        "status": "complete_training_validation_only",
        "test_accessed": False,
        "basis": settings["label"],
        "contract_sha256": file_sha256(CONTRACT),
        "source_manifest_sha256": file_sha256(MANIFEST),
        "budget": {
            "starts": [row["name"] for row in settings["starts"]],
            "residual_bounds": list(RESIDUAL_BOUNDS),
            "steps_per_primary_stage": STEPS_PER_STAGE,
            "training_batches": len(training_batches),
            "validation_batches": len(validation_batches),
            "alternate_activated": bool(alternate),
        },
        "training_first_segmentation_anchor": anchor_training[
            "first_segmentation_loss"
        ],
        "training_first_segmentation_limit": first_limit,
        "soft_first_segmentation_maximum": settings["soft_maximum"],
        "primary_runs": primary,
        "primary_soft_eligible_names": [row["name"] for row in primary_eligible],
        "alternate_runs": alternate,
        "selection_rows": selection_rows,
        "selection": selection,
        "selected_for_derivative_piecewise_and_hard": (
            [] if selection is None else selection["selected_names"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "basis": settings["label"],
                "primary_eligible": len(primary_eligible),
                "alternate_activated": bool(alternate),
                "selected": report["selected_for_derivative_piecewise_and_hard"],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
