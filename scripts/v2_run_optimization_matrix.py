"""Run the bounded validation-only weight sweep and matched v2 design matrix."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from tessscope.optics.model import phase_coefficients
from tessscope.v2.optics.model import unconstrained_from_coefficients
from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    collect_patches,
    joint_value,
    joint_value_and_gradient,
    materialize_joint_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "v2" / "optimization"
SUMMARY = OUTPUT_ROOT / "summary.json"
ASTIGMATIC_REPORT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "gate2" / "astigmatic-autofocus.json"
)
FOCUS_WEIGHT_GRID = (0.01, 0.03, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--pilot-steps", type=int, default=12)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


def coefficient_list(parameters: np.ndarray) -> list[float]:
    return np.asarray(phase_coefficients(jnp.asarray(parameters))).tolist()


def average_validation(services, parameters, batches, calibration, weights):
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


def write_run(name: str, payload: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def adam_design(
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
    gradient_mode="exact",
):
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
            gradient_mode=gradient_mode,
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        parameters -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        row = {
            "step": step,
            "training_objective": value,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "parameters": parameters.tolist(),
            "phase_coefficients": coefficient_list(parameters),
            **branches,
        }
        trace.append(row)
        print(
            json.dumps(
                {
                    "design": name,
                    "step": step,
                    "objective": round(value, 6),
                    "gradient_norm": round(row["gradient_norm"], 6),
                }
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    validation = average_validation(
        services, parameters, validation_batches, calibration, weights
    )
    payload = {
        "name": name,
        "optimizer": "Adam",
        "gradient_mode": gradient_mode,
        "weights": asdict(weights),
        "steps": steps,
        "learning_rate": learning_rate,
        "elapsed_seconds": elapsed,
        "final_parameters": parameters.tolist(),
        "final_phase_coefficients": coefficient_list(parameters),
        "validation": validation,
        "trace": trace,
    }
    write_run(name, payload)
    return payload


def spsa_design(
    services,
    training_batches,
    validation_batches,
    calibration,
    weights,
    *,
    wall_budget_seconds,
    maximum_evaluations,
):
    parameters = np.zeros(6, dtype=np.float32)
    generator = np.random.default_rng(59)
    trace = []
    evaluations = 0
    started = time.perf_counter()
    iteration = 0
    while evaluations + 2 <= maximum_evaluations:
        if time.perf_counter() - started >= wall_budget_seconds:
            break
        iteration += 1
        direction = generator.choice((-1.0, 1.0), size=6).astype(np.float32)
        delta = 0.05 / (iteration**0.101)
        learning_rate = 0.02 / (iteration**0.602)
        batch = training_batches[(iteration - 1) % len(training_batches)]
        plus, _ = joint_value(
            *services,
            parameters + delta * direction,
            batch,
            calibration,
            weights,
        )
        minus, _ = joint_value(
            *services,
            parameters - delta * direction,
            batch,
            calibration,
            weights,
        )
        evaluations += 2
        estimate = (plus - minus) / (2.0 * delta) * direction
        parameters -= learning_rate * estimate
        trace.append(
            {
                "iteration": iteration,
                "evaluations": evaluations,
                "plus": plus,
                "minus": minus,
                "estimate_norm": float(np.linalg.norm(estimate)),
                "parameters": parameters.tolist(),
            }
        )
        print(
            json.dumps(
                {
                    "design": "derivative_free_spsa",
                    "iteration": iteration,
                    "evaluations": evaluations,
                }
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    validation = average_validation(
        services, parameters, validation_batches, calibration, weights
    )
    payload = {
        "name": "derivative_free_spsa",
        "optimizer": "SPSA with two forward evaluations per iteration",
        "weights": asdict(weights),
        "wall_budget_seconds": wall_budget_seconds,
        "maximum_evaluations": maximum_evaluations,
        "elapsed_seconds": elapsed,
        "evaluations": evaluations,
        "final_parameters": parameters.tolist(),
        "final_phase_coefficients": coefficient_list(parameters),
        "validation": validation,
        "trace": trace,
    }
    write_run("derivative-free-spsa", payload)
    return payload


def main() -> None:
    args = parse_args()
    if SUMMARY.exists():
        raise SystemExit(f"Optimization summary already exists: {SUMMARY}")
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
        materialize_joint_batch(training_patches[index : index + 3], noise_seed=1000 + index)
        for index in range(0, len(training_patches), 3)
    ]
    validation_batches = [
        materialize_joint_batch(validation_patches[index : index + 3], noise_seed=None)
        for index in range(0, len(validation_patches), 3)
    ]
    zero = np.zeros(6, dtype=np.float32)
    clear_validation = average_validation(
        services,
        zero,
        validation_batches,
        calibration,
        JointObjectiveWeights(),
    )

    pilots = []
    for focus_weight in FOCUS_WEIGHT_GRID:
        weights = JointObjectiveWeights(focus=focus_weight)
        pilot = adam_design(
            f"pilot-focus-{focus_weight:.2f}",
            services,
            zero,
            training_batches,
            validation_batches,
            calibration,
            weights,
            steps=args.pilot_steps,
            learning_rate=args.learning_rate,
        )
        pilot["normalized_validation_utility"] = (
            pilot["validation"]["segmentation_loss"]
            / clear_validation["segmentation_loss"]
            + pilot["validation"]["focus_mse"] / clear_validation["focus_mse"]
        )
        pilots.append(pilot)
    selected_pilot = min(pilots, key=lambda row: row["normalized_validation_utility"])
    joint_weights = JointObjectiveWeights(
        focus=float(selected_pilot["weights"]["focus"])
    )

    segmentation = adam_design(
        "segmentation-only",
        services,
        zero,
        training_batches,
        validation_batches,
        calibration,
        JointObjectiveWeights(segmentation=1.0, focus=0.0),
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    focus = adam_design(
        "focus-only",
        services,
        zero,
        training_batches,
        validation_batches,
        calibration,
        JointObjectiveWeights(segmentation=0.0, focus=1.0),
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    joint_starts = [
        adam_design(
            f"exact-joint-start-{start_index}",
            services,
            (
                zero
                if start_index == 0
                else np.random.default_rng(61).normal(0, 0.05, size=6).astype(np.float32)
            ),
            training_batches,
            validation_batches,
            calibration,
            joint_weights,
            steps=args.steps,
            learning_rate=args.learning_rate,
        )
        for start_index in range(2)
    ]
    selected_joint = min(
        joint_starts,
        key=lambda row: (
            row["validation"]["segmentation_loss"] / clear_validation["segmentation_loss"]
            + row["validation"]["focus_mse"] / clear_validation["focus_mse"]
        ),
    )
    broken = adam_design(
        "broken-focus-gradient",
        services,
        zero,
        training_batches,
        validation_batches,
        calibration,
        joint_weights,
        steps=args.steps,
        learning_rate=args.learning_rate,
        gradient_mode="stop_focus",
    )

    segmentation_coefficients = np.asarray(segmentation["final_phase_coefficients"])
    focus_coefficients = np.asarray(focus["final_phase_coefficients"])
    superposed_coefficients = segmentation_coefficients + focus_coefficients
    superposed_norm = float(np.linalg.norm(superposed_coefficients))
    if superposed_norm >= 2.5:
        superposed_coefficients *= 2.5 * 0.999 / superposed_norm
    superposed_parameters = unconstrained_from_coefficients(superposed_coefficients)
    superposed_validation = average_validation(
        services,
        superposed_parameters,
        validation_batches,
        calibration,
        joint_weights,
    )
    superposed = {
        "name": "naive-superposition",
        "source_designs": ["segmentation-only", "focus-only"],
        "final_parameters": superposed_parameters.tolist(),
        "final_phase_coefficients": superposed_coefficients.tolist(),
        "validation": superposed_validation,
    }
    write_run("naive-superposition", superposed)

    astigmatic = json.loads(ASTIGMATIC_REPORT.read_text())["selected"]
    astigmatic_parameters = np.asarray(astigmatic["phase_parameters"], dtype=np.float32)
    astigmatic_design = {
        "name": "classical-astigmatic",
        "selection_source": str(ASTIGMATIC_REPORT.relative_to(PROJECT_ROOT)),
        "final_parameters": astigmatic_parameters.tolist(),
        "final_phase_coefficients": coefficient_list(astigmatic_parameters),
        "validation": average_validation(
            services,
            astigmatic_parameters,
            validation_batches,
            calibration,
            joint_weights,
        ),
    }
    write_run("classical-astigmatic", astigmatic_design)

    derivative_free = spsa_design(
        services,
        training_batches,
        validation_batches,
        calibration,
        joint_weights,
        wall_budget_seconds=float(selected_joint["elapsed_seconds"]),
        maximum_evaluations=2 * args.steps,
    )
    summary = {
        "status": "complete_training_validation_only",
        "test_accessed": False,
        "calibration": asdict(calibration),
        "budget": {
            "pilot_steps": args.pilot_steps,
            "promoted_steps": args.steps,
            "learning_rate": args.learning_rate,
            "training_batches": len(training_batches),
            "validation_batches": len(validation_batches),
        },
        "clear_validation": clear_validation,
        "weight_pilots": [
            {
                "name": row["name"],
                "focus_weight": row["weights"]["focus"],
                "validation": row["validation"],
                "normalized_validation_utility": row[
                    "normalized_validation_utility"
                ],
            }
            for row in pilots
        ],
        "selected_joint_weights": asdict(joint_weights),
        "designs": {
            "segmentation_only": segmentation,
            "focus_only": focus,
            "exact_joint_starts": joint_starts,
            "selected_exact_joint": selected_joint["name"],
            "broken_focus_gradient": broken,
            "naive_superposition": superposed,
            "classical_astigmatic": astigmatic_design,
            "derivative_free": derivative_free,
            "clear": {
                "final_parameters": zero.tolist(),
                "final_phase_coefficients": coefficient_list(zero),
                "validation": clear_validation,
            },
            "cubic": {
                "status": "pending matched validation selection in cubic phase family"
            },
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(SUMMARY),
                "selected_focus_weight": joint_weights.focus,
                "selected_exact_joint": selected_joint["name"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
