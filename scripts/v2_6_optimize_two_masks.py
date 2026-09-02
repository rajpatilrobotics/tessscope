"""Run frozen feasibility restoration and the bounded exact two-mask matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration
from tessscope.v2_5.constrained import update_duals
from tessscope.v2_5.evaluation import parameter_sha256, physical_diagnostics
from tessscope.v2_6.data import materialize_two_mask_batches
from tessscope.v2_6.optimization import (
    constrained_value_and_gradient,
    feasibility_value_and_gradient,
    feasibility_violations,
    project_two_masks,
)
from tessscope.v2_6.selection import select_soft_rows, soft_eligible
from tessscope.v2_6.two_mask import (
    average_two_mask_forward,
    join_two_mask_parameters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT_ROOT / "configs" / "v2_6" / "two-mask-contract.yaml"
SOURCES = PROJECT_ROOT / "configs" / "v2_6" / "two-mask-sources.json"
BASELINES = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "two-mask"
    / "matched-baseline-screen.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "two-mask"
    / "exact-optimization.json"
)
CHECKPOINT_ROOT = PROJECT_ROOT / "artifacts" / "runtime-runs" / "v2_6-two-mask"
RESIDUAL_BOUNDS = (0.075, 0.055)
RESTORATION_STEPS = 8
PRIMARY_STEPS = 18
RESTORATION_RATE = 0.008
PRIMARY_RATE = 0.006


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8407")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adam_update(
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    *,
    step: int,
    rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = 0.9 * first_moment + 0.1 * gradient
    second = 0.999 * second_moment + 0.001 * np.square(gradient)
    corrected_first = first / (1.0 - 0.9**step)
    corrected_second = second / (1.0 - 0.999**step)
    update = rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return update, first, second


def _pair_physical(parameters: np.ndarray) -> dict:
    return {
        "sensing": physical_diagnostics(parameters[:7], basis_size=7),
        "capture": physical_diagnostics(parameters[7:], basis_size=7),
    }


def restore_feasibility(
    *,
    name: str,
    parameters: np.ndarray,
    residual_limit: float,
    first_limit: float,
    services,
    batches,
    calibration,
) -> dict:
    label = str(residual_limit).replace(".", "p")
    path = CHECKPOINT_ROOT / f"{name}-restore-{label}.json"
    if path.exists():
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("completed"):
            return checkpoint["result"]
        value = np.asarray(checkpoint["parameters"], dtype=np.float32)
        first_moment = np.asarray(checkpoint["first_moment"], dtype=np.float32)
        second_moment = np.asarray(checkpoint["second_moment"], dtype=np.float32)
        trace = checkpoint["trace"]
        first_step = checkpoint["next_step"]
        elapsed_before = checkpoint["elapsed_seconds"]
    else:
        value = np.asarray(parameters, dtype=np.float32).copy()
        first_moment = np.zeros(7, dtype=np.float32)
        second_moment = np.zeros(7, dtype=np.float32)
        trace = []
        first_step = 1
        elapsed_before = 0.0
    started = time.perf_counter()
    for step in range(first_step, RESTORATION_STEPS + 1):
        objective, gradient, diagnostics = feasibility_value_and_gradient(
            *services,
            value,
            batches,
            calibration,
            first_limit=first_limit,
            residual_limit=residual_limit,
        )
        feasible = bool(
            diagnostics["first_scaled_violation"] <= 1e-4
            and diagnostics["residual_scaled_violation"] <= 1e-4
        )
        update = np.zeros(14, dtype=np.float32)
        if not feasible:
            sensing_update, first_moment, second_moment = _adam_update(
                gradient[:7],
                first_moment,
                second_moment,
                step=step,
                rate=RESTORATION_RATE,
            )
            update[:7] = sensing_update
            value -= update
            value, projected = project_two_masks(value)
        else:
            projected = (False, False)
        trace.append(
            {
                "step": step,
                "objective": objective,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "sensing_update_norm": float(np.linalg.norm(update[:7])),
                "parameters_after_step": value.tolist(),
                "projection_applied": list(projected),
                "feasible_before_step": feasible,
                **diagnostics,
            }
        )
        elapsed = elapsed_before + time.perf_counter() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "completed": False,
                    "parameters": value.tolist(),
                    "first_moment": first_moment.tolist(),
                    "second_moment": second_moment.tolist(),
                    "trace": trace,
                    "next_step": step + 1,
                    "elapsed_seconds": elapsed,
                }
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "run": name,
                    "phase": "restore",
                    "bound": residual_limit,
                    "step": step,
                    "objective": round(objective, 6),
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
        if feasible:
            break
    elapsed = elapsed_before + time.perf_counter() - started
    metrics = average_two_mask_forward(services, value, batches, calibration)
    violations = feasibility_violations(
        metrics,
        first_limit=first_limit,
        residual_limit=residual_limit,
    )
    result = {
        "name": name,
        "residual_bound": residual_limit,
        "steps": len(trace),
        "elapsed_seconds": elapsed,
        "final_parameters": value.tolist(),
        "metrics": metrics,
        "scaled_violations": violations.tolist(),
        "feasible": bool(np.max(violations) <= 1e-4),
        "trace": trace,
    }
    path.write_text(json.dumps({"completed": True, "result": result}) + "\n")
    return result


def optimize_stage(
    *,
    name: str,
    parameters: np.ndarray,
    residual_limit: float,
    first_limit: float,
    services,
    batches,
    calibration,
) -> dict:
    label = str(residual_limit).replace(".", "p")
    path = CHECKPOINT_ROOT / f"{name}-primary-{label}.json"
    if path.exists():
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("completed"):
            return checkpoint["result"]
        value = np.asarray(checkpoint["parameters"], dtype=np.float32)
        first_moment = np.asarray(checkpoint["first_moment"], dtype=np.float32)
        second_moment = np.asarray(checkpoint["second_moment"], dtype=np.float32)
        duals = np.asarray(checkpoint["duals"], dtype=np.float32)
        trace = checkpoint["trace"]
        first_step = checkpoint["next_step"]
        elapsed_before = checkpoint["elapsed_seconds"]
    else:
        value = np.asarray(parameters, dtype=np.float32).copy()
        first_moment = np.zeros(14, dtype=np.float32)
        second_moment = np.zeros(14, dtype=np.float32)
        duals = np.zeros(2, dtype=np.float32)
        trace = []
        first_step = 1
        elapsed_before = 0.0
    started = time.perf_counter()
    for step in range(first_step, PRIMARY_STEPS + 1):
        penalty = (4.0, 8.0, 16.0)[min((step - 1) // 6, 2)]
        objective, gradient, diagnostics = constrained_value_and_gradient(
            *services,
            value,
            batches,
            calibration,
            first_limit=first_limit,
            residual_limit=residual_limit,
            duals=duals,
            penalty=penalty,
        )
        rate = PRIMARY_RATE * (0.3 if step > 12 else 1.0)
        update, first_moment, second_moment = _adam_update(
            gradient,
            first_moment,
            second_moment,
            step=step,
            rate=rate,
        )
        value -= update
        value, projected = project_two_masks(value)
        if step % 6 == 0:
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
                "objective": objective,
                "learning_rate": rate,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "update_norm": float(np.linalg.norm(update)),
                "parameters_after_step": value.tolist(),
                "projection_applied": list(projected),
                "duals_after_step": duals.tolist(),
                **diagnostics,
            }
        )
        elapsed = elapsed_before + time.perf_counter() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "completed": False,
                    "parameters": value.tolist(),
                    "first_moment": first_moment.tolist(),
                    "second_moment": second_moment.tolist(),
                    "duals": duals.tolist(),
                    "trace": trace,
                    "next_step": step + 1,
                    "elapsed_seconds": elapsed,
                }
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "run": name,
                    "phase": "primary",
                    "bound": residual_limit,
                    "step": step,
                    "final": round(diagnostics["final_segmentation_loss"], 6),
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
    metrics = average_two_mask_forward(services, value, batches, calibration)
    result = {
        "name": name,
        "residual_bound": residual_limit,
        "steps": PRIMARY_STEPS,
        "elapsed_seconds": elapsed,
        "final_parameters": value.tolist(),
        "parameter_sha256": parameter_sha256(value),
        "metrics": metrics,
        "physical": _pair_physical(value),
        "trace": trace,
    }
    path.write_text(json.dumps({"completed": True, "result": result}) + "\n")
    return result


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"V2.6 two-mask optimization already exists: {OUTPUT}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    optimization_batches = materialize_two_mask_batches("optimization")
    development_batches = materialize_two_mask_batches("development")
    confirmation_batches = materialize_two_mask_batches("confirmation")
    sources = json.loads(SOURCES.read_text())
    baselines = json.loads(BASELINES.read_text())
    segmentation = sources["parameters"]["b7_segmentation_only"]
    starts = {
        "piecewise_sensing_to_segmentation_capture": join_two_mask_parameters(
            sources["parameters"]["v2_2_piecewise_028"], segmentation
        ),
        "balanced_sensing_to_segmentation_capture": join_two_mask_parameters(
            sources["parameters"]["v2_4_balanced_step14"], segmentation
        ),
    }
    anchor_parameters = join_two_mask_parameters(segmentation, segmentation)
    anchors = {
        partition: average_two_mask_forward(
            services, anchor_parameters, batches, calibration
        )
        for partition, batches in (
            ("optimization", optimization_batches),
            ("development", development_batches),
            ("confirmation", confirmation_batches),
        )
    }
    optimization_first_limit = (
        anchors["optimization"]["first_segmentation_loss"] + 0.008
    )
    endpoints = []
    restorations = []
    for start_name, initial in starts.items():
        parameters = initial.copy()
        for residual_bound in RESIDUAL_BOUNDS:
            restoration = restore_feasibility(
                name=start_name,
                parameters=parameters,
                residual_limit=residual_bound,
                first_limit=optimization_first_limit,
                services=services,
                batches=optimization_batches,
                calibration=calibration,
            )
            restorations.append(restoration)
            stage = optimize_stage(
                name=start_name,
                parameters=np.asarray(
                    restoration["final_parameters"], dtype=np.float32
                ),
                residual_limit=residual_bound,
                first_limit=optimization_first_limit,
                services=services,
                batches=optimization_batches,
                calibration=calibration,
            )
            parameters = np.asarray(stage["final_parameters"], dtype=np.float32)
            development = average_two_mask_forward(
                services, parameters, development_batches, calibration
            )
            endpoints.append(
                {
                    **stage,
                    "start_name": start_name,
                    "development_metrics": development,
                }
            )

    baseline_name = baselines["best_confirmation_baseline_name"]
    baseline_confirmation = next(
        (
            row
            for row in baselines["confirmation_rows"]
            if row["name"] == baseline_name
        ),
        None,
    )
    baseline_development_limit = (
        None
        if baseline_confirmation is None
        else baseline_confirmation["development_metrics"]["final_segmentation_loss"]
    )
    selection_rows = []
    for endpoint in endpoints:
        metrics = endpoint["development_metrics"]
        eligible_without_baseline = soft_eligible(
            metrics,
            endpoint["physical"],
            first_limit=anchors["development"]["first_segmentation_loss"] + 0.008,
            residual_limit=endpoint["residual_bound"],
            baseline_final_limit=baseline_development_limit,
        )
        selection_rows.append(
            {
                "name": endpoint["name"]
                + f"-residual-{endpoint['residual_bound']:.3f}",
                "endpoint_name": endpoint["name"],
                "start_name": endpoint["start_name"],
                "residual_bound": endpoint["residual_bound"],
                "parameters": endpoint["final_parameters"],
                "parameter_sha256": endpoint["parameter_sha256"],
                "metrics": metrics,
                "physical": endpoint["physical"],
                "eligible_without_matched_baseline": eligible_without_baseline,
                "matched_baseline_available": baseline_confirmation is not None,
                "eligible": bool(
                    eligible_without_baseline and baseline_confirmation is not None
                ),
            }
        )
    selected = select_soft_rows(selection_rows, maximum=2)
    selection_by_name = {row["name"]: row for row in selection_rows}
    confirmation_rows = []
    for name in selected:
        row = selection_by_name[name]
        metrics = average_two_mask_forward(
            services,
            np.asarray(row["parameters"], dtype=np.float32),
            confirmation_batches,
            calibration,
        )
        eligible = soft_eligible(
            metrics,
            row["physical"],
            first_limit=anchors["confirmation"]["first_segmentation_loss"] + 0.008,
            residual_limit=row["residual_bound"],
            baseline_final_limit=baseline_confirmation["metrics"][
                "final_segmentation_loss"
            ],
        )
        confirmation_rows.append({**row, "metrics": metrics, "eligible": eligible})
    confirmed = select_soft_rows(confirmation_rows, maximum=1)
    result = {
        "status": (
            "complete_training_only_candidate_selected"
            if confirmed
            else "complete_training_only_negative"
        ),
        "test_accessed": False,
        "contract_sha256": _sha256(CONTRACT),
        "sources_sha256": _sha256(SOURCES),
        "baseline_screen_sha256": _sha256(BASELINES),
        "budget": {
            "starts": list(starts),
            "residual_bounds": list(RESIDUAL_BOUNDS),
            "restoration_steps_maximum_per_stage": RESTORATION_STEPS,
            "primary_steps_per_stage": PRIMARY_STEPS,
            "primary_steps_maximum": len(starts)
            * len(RESIDUAL_BOUNDS)
            * PRIMARY_STEPS,
            "optimization_batches": len(optimization_batches),
            "development_batches": len(development_batches),
            "confirmation_batches": len(confirmation_batches),
        },
        "anchors": anchors,
        "optimization_first_limit": optimization_first_limit,
        "matched_baseline_name": baseline_name,
        "matched_baseline_available": baseline_confirmation is not None,
        "matched_baseline_confirmation": baseline_confirmation,
        "restorations": restorations,
        "endpoints": endpoints,
        "development_selection_rows": selection_rows,
        "selected_for_confirmation": selected,
        "confirmation_rows": confirmation_rows,
        "selected_for_hard_validation": confirmed,
        "stop_reason": (
            None
            if confirmed
            else (
                "no_eligible_matched_two_mask_baseline"
                if baseline_confirmation is None
                else "no_exact_candidate_passed_confirmation_gates"
            )
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "matched_baseline_available": result[
                    "matched_baseline_available"
                ],
                "eligible_without_matched_baseline": [
                    row["name"]
                    for row in selection_rows
                    if row["eligible_without_matched_baseline"]
                ],
                "selected_for_confirmation": selected,
                "selected_for_hard_validation": confirmed,
                "stop_reason": result["stop_reason"],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
