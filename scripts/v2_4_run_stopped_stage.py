"""Run matched stopped-stage-gradient controls for frozen v2.4 checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration, collect_patches
from tessscope.v2_3.closed_loop import (
    OBJECTIVE_PROFILES,
    closed_loop_value,
    closed_loop_value_and_gradient,
    materialize_closed_loop_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION = PROJECT_ROOT / "configs" / "v2_4" / "selected-checkpoints.json"
V2_3_MATRIX = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_3"
    / "optimization"
    / "closed-loop-matrix.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "stopped-stage-matched.json"
)
CACHE_ROOT = PROJECT_ROOT / "artifacts" / "runtime-runs" / "v2_4-stopped-stage"
LEARNING_RATE = 0.01
FORWARD_PARITY_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8407")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def average_validation(
    services, parameters, batches, calibration, weights, *, gradient_mode
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


def optimize_stopped(
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
    steps,
) -> dict:
    checkpoint_path = CACHE_ROOT / f"{name}.json"
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
        gradient_mode="stop_stage",
    )
    started = time.perf_counter()
    for step in range(first_step, steps + 1):
        batch = training_batches[(step - 1) % len(training_batches)]
        value, gradient, diagnostics = closed_loop_value_and_gradient(
            *services,
            parameters,
            batch,
            calibration,
            weights,
            gradient_mode="stop_stage",
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
                    "of": steps,
                    "objective": round(value, 6),
                    "residual_mae_um": round(diagnostics["residual_mae_um"], 6),
                }
            ),
            flush=True,
        )
    elapsed = elapsed_before + time.perf_counter() - started
    stopped_validation = average_validation(
        services,
        parameters,
        validation_batches,
        calibration,
        weights,
        gradient_mode="stop_stage",
    )
    exact_validation = average_validation(
        services,
        parameters,
        validation_batches,
        calibration,
        weights,
        gradient_mode="exact",
    )
    parity = {
        key: abs(stopped_validation[key] - exact_validation[key])
        for key in stopped_validation
    }
    result = {
        "name": name,
        "profile": profile,
        "start_name": start_name,
        "gradient_mode": "stop_stage",
        "steps": steps,
        "initial_learning_rate": LEARNING_RATE,
        "elapsed_seconds": elapsed,
        "start_parameters": np.asarray(start_parameters).tolist(),
        "final_parameters": parameters.tolist(),
        "start_validation": start_validation,
        "stopped_validation": stopped_validation,
        "exact_forward_validation": exact_validation,
        "forward_parity_absolute_by_metric": parity,
        "maximum_forward_parity_absolute": max(parity.values()),
        "trace": trace,
    }
    checkpoint_path.write_text(
        json.dumps({"completed": True, "result": result}) + "\n"
    )
    return result


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"V2.4 stopped-stage artifact already exists: {OUTPUT}")
    selection = json.loads(SELECTION.read_text())
    matrix = json.loads(V2_3_MATRIX.read_text())
    run_by_name = {run["name"]: run for run in matrix["exact_runs"]}
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
            training_patches[index : index + 3], noise_seed=4100 + index
        )
        for index in range(0, len(training_patches), 3)
    ]
    validation_batches = [
        materialize_closed_loop_batch(
            validation_patches[index : index + 3], noise_seed=None
        )
        for index in range(0, len(validation_patches), 3)
    ]
    pairs = []
    for exact in selection["selected"]:
        source_run = run_by_name[exact["source_run"]]
        stopped_name = (
            f"v2_4-stopped-{exact['profile']}-{exact['start_name']}"
            f"-step-{exact['step']:02d}"
        )
        stopped = optimize_stopped(
            name=stopped_name,
            services=services,
            start_parameters=source_run["start_parameters"],
            training_batches=training_batches,
            validation_batches=validation_batches,
            calibration=calibration,
            weights=OBJECTIVE_PROFILES[exact["profile"]],
            profile=exact["profile"],
            start_name=exact["start_name"],
            steps=exact["step"],
        )
        if stopped["maximum_forward_parity_absolute"] > FORWARD_PARITY_TOLERANCE:
            raise SystemExit(f"Forward parity failed for {stopped_name}")
        pairs.append({"exact": exact, "stopped": stopped})
    report = {
        "status": "complete_matched_stopped_stage",
        "test_accessed": False,
        "hard_labels_accessed": False,
        "forward_parity_tolerance": FORWARD_PARITY_TOLERANCE,
        "maximum_forward_parity_absolute": max(
            pair["stopped"]["maximum_forward_parity_absolute"] for pair in pairs
        ),
        "pairs": pairs,
        "next_gate": "expanded_hard_validation",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "pairs": len(pairs),
                "maximum_forward_parity_absolute": report[
                    "maximum_forward_parity_absolute"
                ],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
