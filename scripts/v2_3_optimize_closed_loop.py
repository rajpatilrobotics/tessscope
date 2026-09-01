"""Run the pre-registered exact v2.3 profile/start matrix and stopped ablation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration, collect_patches
from tessscope.v2_3.closed_loop import (
    OBJECTIVE_PROFILES,
    closed_loop_value,
    closed_loop_value_and_gradient,
    materialize_closed_loop_batch,
    nondominated_closed_loop_names,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEPARATE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-separate-baselines.json"
)
JOINT = PROJECT_ROOT / "configs" / "v2_2" / "frozen-joint-candidate.json"
PIECEWISE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "screening"
    / "piecewise-soft-frontier.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_3"
    / "optimization"
    / "closed-loop-matrix.json"
)
CHECKPOINT_ROOT = PROJECT_ROOT / "artifacts" / "runtime-runs" / "v2_3-optimization"
STEPS = 30
LEARNING_RATE = 0.01
FIRST_SEGMENTATION_MAXIMUM = 1.090270310640335 + 0.008


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8407")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def average_validation(
    services,
    parameters,
    batches,
    calibration,
    weights,
    *,
    gradient_mode,
) -> dict:
    rows = []
    for batch in batches:
        value, diagnostics = closed_loop_value(
            *services,
            parameters,
            batch,
            calibration,
            weights,
            gradient_mode=gradient_mode,
        )
        rows.append({"objective": value, **diagnostics})
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def optimize(
    *,
    name,
    services,
    start_parameters,
    training_batches,
    validation_batches,
    calibration,
    weights,
    profile,
    start_name,
    gradient_mode,
) -> dict:
    checkpoint_path = CHECKPOINT_ROOT / f"{name}.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("completed"):
            return checkpoint["result"]
        parameters = np.asarray(checkpoint["parameters"], dtype=np.float32)
        first_moment = np.asarray(checkpoint["first_moment"], dtype=np.float32)
        second_moment = np.asarray(checkpoint["second_moment"], dtype=np.float32)
        trace = checkpoint["trace"]
        first_step = int(checkpoint["next_step"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
    else:
        parameters = np.asarray(start_parameters, dtype=np.float32).copy()
        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        trace = []
        first_step = 1
        elapsed_before = 0.0
    start_validation = average_validation(
        services,
        start_parameters,
        validation_batches,
        calibration,
        weights,
        gradient_mode=gradient_mode,
    )
    started = time.perf_counter()
    for step in range(first_step, STEPS + 1):
        batch = training_batches[(step - 1) % len(training_batches)]
        value, gradient, diagnostics = closed_loop_value_and_gradient(
            *services,
            parameters,
            batch,
            calibration,
            weights,
            gradient_mode=gradient_mode,
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        learning_rate = LEARNING_RATE * (0.3 if step > 20 else 1.0)
        parameters -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        trace.append(
            {
                "step": step,
                "learning_rate": learning_rate,
                "training_objective": value,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "parameters": parameters.tolist(),
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
                    "design": name,
                    "step": step,
                    "objective": round(value, 6),
                    "residual_mae_um": round(diagnostics["residual_mae_um"], 5),
                }
            ),
            flush=True,
        )
    elapsed = elapsed_before + time.perf_counter() - started
    validation = average_validation(
        services,
        parameters,
        validation_batches,
        calibration,
        weights,
        gradient_mode=gradient_mode,
    )
    result = {
        "name": name,
        "profile": profile,
        "start_name": start_name,
        "gradient_mode": gradient_mode,
        "weights": asdict(weights),
        "steps": STEPS,
        "initial_learning_rate": LEARNING_RATE,
        "elapsed_seconds": elapsed,
        "start_parameters": np.asarray(start_parameters).tolist(),
        "final_parameters": parameters.tolist(),
        "start_validation": start_validation,
        "validation": validation,
        "trace": trace,
    }
    checkpoint_path.write_text(
        json.dumps({"completed": True, "result": result}) + "\n"
    )
    return result


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"V2.3 optimization artifact already exists: {OUTPUT}")
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
        materialize_closed_loop_batch(
            training_patches[index : index + 3],
            noise_seed=4100 + index,
        )
        for index in range(0, len(training_patches), 3)
    ]
    validation_batches = [
        materialize_closed_loop_batch(
            validation_patches[index : index + 3], noise_seed=None
        )
        for index in range(0, len(validation_patches), 3)
    ]
    separate = json.loads(SEPARATE.read_text())
    joint = json.loads(JOINT.read_text())
    piecewise = json.loads(PIECEWISE.read_text())
    piecewise_028 = next(
        point for point in piecewise["points"] if point["name"] == "v2_2-piecewise-028"
    )
    starts = {
        "segmentation_only": separate["segmentation_only"]["final_parameters"],
        "projected_joint": joint["parameters"],
        "piecewise_028": piecewise_028["parameters"],
    }
    exact_runs = []
    for profile, weights in OBJECTIVE_PROFILES.items():
        for start_name, start_parameters in starts.items():
            name = f"v2_3-exact-{profile}-{start_name}"
            exact_runs.append(
                optimize(
                    name=name,
                    services=services,
                    start_parameters=start_parameters,
                    training_batches=training_batches,
                    validation_batches=validation_batches,
                    calibration=calibration,
                    weights=weights,
                    profile=profile,
                    start_name=start_name,
                    gradient_mode="exact",
                )
            )
    eligible = [
        run
        for run in exact_runs
        if run["validation"]["first_segmentation_loss"]
        <= FIRST_SEGMENTATION_MAXIMUM
        and run["validation"]["residual_mae_um"]
        < run["start_validation"]["residual_mae_um"]
    ]
    selection_rows = [
        {"name": run["name"], **run["validation"]} for run in eligible
    ]
    nondominated = nondominated_closed_loop_names(selection_rows)
    by_name = {run["name"]: run for run in exact_runs}
    selected = sorted(
        nondominated,
        key=lambda name: (
            by_name[name]["validation"]["final_segmentation_loss"],
            by_name[name]["validation"]["residual_mae_um"],
            by_name[name]["validation"]["first_segmentation_loss"],
            name,
        ),
    )[:3]
    if not selected:
        raise SystemExit("No exact v2.3 endpoint passed the frozen soft gate")
    best = by_name[selected[0]]
    stopped_name = f"v2_3-stopped-{best['profile']}-{best['start_name']}"
    stopped = optimize(
        name=stopped_name,
        services=services,
        start_parameters=starts[best["start_name"]],
        training_batches=training_batches,
        validation_batches=validation_batches,
        calibration=calibration,
        weights=OBJECTIVE_PROFILES[best["profile"]],
        profile=best["profile"],
        start_name=best["start_name"],
        gradient_mode="stop_stage",
    )
    report = {
        "status": "complete_training_validation_only",
        "test_accessed": False,
        "budget": {
            "profiles": list(OBJECTIVE_PROFILES),
            "starts": list(starts),
            "steps_per_run": STEPS,
            "training_batches": len(training_batches),
            "validation_batches": len(validation_batches),
        },
        "first_segmentation_maximum": FIRST_SEGMENTATION_MAXIMUM,
        "exact_runs": exact_runs,
        "soft_eligible_exact_names": [run["name"] for run in eligible],
        "soft_nondominated_exact_names": sorted(nondominated),
        "selected_for_hard_validation": selected,
        "stopped_stage_ablation": stopped,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selected_for_hard_validation": selected,
                "stopped_stage_ablation": stopped_name,
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
