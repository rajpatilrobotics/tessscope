"""Run the three one-start learned designs under the frozen matched budget."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from tesseract_core import Tesseract

from tessscope.data.bbbc039 import (
    BIOLOGICAL_SAMPLING_UM,
    OBSERVER_SAMPLING_UM,
    ObjectNormalization,
    patch_origins,
    prepare_patch,
    records_for_split,
)
from tessscope.evaluation.metrics import instance_metrics, valid_region_labels
from tessscope.evaluation.route import deterministic_sensor, segment_sensor_batch
from tessscope.observer.calibration import load_frozen_transform
from tessscope.observer.instanseg import load_frozen_model
from tessscope.optics.model import (
    phase_coefficients,
    psf_support_energy_fraction,
    simulate_sensor,
)
from tessscope.optimization.adam import AdamState, adam_update
from tessscope.optimization.schedule import build_training_schedule
from tessscope.optimization.training import (
    image_value_and_gradient,
    materialize_batch,
    task_value_and_gradient,
)

STEPS = 120
LEARNING_RATE = 0.01
SURROGATE_SCALE = 108.97649746350436
DEPTHS_UM = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
PRIMARY_DEPTHS_UM = {-6.0, -4.0, -2.0, 2.0, 4.0, 6.0}
CLEAR_VALIDATION_OFF_FOCUS_PQ = 0.56355035914339


@dataclass(frozen=True)
class ValidationScore:
    mean_off_focus_pq: float
    worst_depth_pq: float
    focus_pq: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8401")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/gate8")
    )
    return parser.parse_args()


def clear_focus_reference(object_batch: np.ndarray) -> np.ndarray:
    reference = simulate_sensor(
        jnp.zeros((6,), dtype=jnp.float32),
        jnp.asarray(object_batch, dtype=jnp.float32),
        jnp.asarray([0.0], dtype=jnp.float32),
    )
    return np.maximum(np.asarray(reference.block_until_ready()), 0.0)


def run_design(
    name: str,
    optics: Tesseract,
    observer: Tesseract,
    output_dir: Path,
) -> dict[str, object]:
    parameters = np.zeros((6,), dtype=np.float32)
    state = AdamState.zeros(parameters.shape)
    trace: list[dict[str, object]] = []
    started = time.perf_counter()
    for spec in build_training_schedule(steps=STEPS):
        batch = materialize_batch(spec)
        if name == "image_fidelity":
            value, gradient = image_value_and_gradient(
                optics,
                parameters,
                batch,
                clear_focus_reference(batch.object_batch),
            )
        else:
            value, gradient = task_value_and_gradient(
                optics,
                observer,
                parameters,
                batch,
                backward_mode="exact" if name == "exact_task" else "surrogate",
                surrogate_scale=SURROGATE_SCALE,
            )
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError(f"Non-finite {name} result at step {spec.step}")
        parameters, state = adam_update(
            parameters,
            gradient,
            state,
            learning_rate=LEARNING_RATE,
        )
        trace.append(
            {
                "step": spec.step,
                "loss": value,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "parameters": parameters.tolist(),
                "patch_ids": list(batch.patch_ids),
                "depths_um": batch.depths_um.tolist(),
            }
        )
        if spec.step % 10 == 0 or spec.step == STEPS - 1:
            print(
                json.dumps(
                    {
                        "design": name,
                        "step": spec.step,
                        "loss": value,
                        "gradient_norm": float(np.linalg.norm(gradient)),
                    }
                ),
                flush=True,
            )
    coefficients = np.asarray(phase_coefficients(jnp.asarray(parameters)))
    result: dict[str, object] = {
        "design": name,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "start_parameters": [0.0] * 6,
        "final_parameters": parameters.tolist(),
        "final_phase_coefficients": coefficients.tolist(),
        "final_phase_rms_radians": float(np.linalg.norm(coefficients)),
        "wall_seconds": time.perf_counter() - started,
        "trace": trace,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}-training.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def validation_score(
    parameters: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> ValidationScore:
    normalization = ObjectNormalization(125.0, 1642.0)
    transform = load_frozen_transform()
    margin = int(
        np.floor(48 * BIOLOGICAL_SAMPLING_UM / OBSERVER_SAMPLING_UM + 0.5)
    )
    rows: list[tuple[float, float]] = []
    for record in records_for_split("validation"):
        for origin in patch_origins("validation"):
            patch = prepare_patch(record, origin, normalization)
            sensor = deterministic_sensor(parameters, patch.object_image, DEPTHS_UM)
            predictions = segment_sensor_batch(
                model,
                sensor,
                patch.instance_labels.shape,
                device=device,
                maximum_batch_size=4,
                transform=transform,
            )
            for depth, prediction in zip(DEPTHS_UM, predictions, strict=True):
                target, valid_prediction = valid_region_labels(
                    patch.instance_labels, prediction, margin
                )
                rows.append(
                    (
                        float(depth),
                        instance_metrics(target, valid_prediction).panoptic_quality,
                    )
                )
    by_depth = {
        float(depth): float(np.mean([pq for row_depth, pq in rows if row_depth == depth]))
        for depth in DEPTHS_UM
    }
    primary = [by_depth[depth] for depth in sorted(PRIMARY_DEPTHS_UM)]
    return ValidationScore(
        mean_off_focus_pq=float(np.mean(primary)),
        worst_depth_pq=float(min(primary)),
        focus_pq=by_depth[0.0],
    )


def support_report(parameters: np.ndarray) -> dict[str, object]:
    by_depth = {
        str(float(depth)): float(
            psf_support_energy_fraction(
                jnp.asarray(parameters, dtype=jnp.float32),
                jnp.asarray(depth, dtype=jnp.float32),
            )
        )
        for depth in DEPTHS_UM
    }
    minimum = min(by_depth.values())
    return {
        "by_depth": by_depth,
        "minimum": minimum,
        "required_minimum": 0.995,
        "passed": bool(minimum >= 0.995),
    }


def main() -> None:
    args = parse_args()
    optics = Tesseract.from_url(args.optics_url, timeout=120)
    observer = Tesseract.from_url(args.observer_url, timeout=120)
    optics.health()
    observer.health()
    results = [
        run_design(name, optics, observer, args.output_dir)
        for name in ("exact_task", "surrogate", "image_fidelity")
    ]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    summaries = []
    for result in results:
        parameters = np.asarray(result["final_parameters"], dtype=np.float32)
        validation = validation_score(parameters, model, device)
        support = support_report(parameters)
        result["validation"] = asdict(validation)
        result["support_energy_fraction"] = support
        (args.output_dir / f"{result['design']}-training.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        summaries.append(
            {
                "design": result["design"],
                "final_parameters": result["final_parameters"],
                "final_phase_coefficients": result["final_phase_coefficients"],
                "final_phase_rms_radians": result["final_phase_rms_radians"],
                "wall_seconds": result["wall_seconds"],
                "validation": asdict(validation),
                "support_energy_fraction": support,
            }
        )
        print(json.dumps(summaries[-1]), flush=True)

    exact = next(summary for summary in summaries if summary["design"] == "exact_task")
    support_passed = all(
        summary["support_energy_fraction"]["passed"] for summary in summaries
    )
    validation_improved = (
        exact["validation"]["mean_off_focus_pq"] > CLEAR_VALIDATION_OFF_FOCUS_PQ
    )
    report = {
        "gate": "gate8_matched_one_start_designs",
        "budget": {
            "start": "shared zero",
            "steps_per_design": STEPS,
            "batch_size": 2,
            "depths_per_step": 3,
            "logical_sample_depths_per_design": STEPS * 2 * 3,
            "learning_rate": LEARNING_RATE,
        },
        "surrogate_frozen_scale": SURROGATE_SCALE,
        "clear_validation_mean_off_focus_pq": CLEAR_VALIDATION_OFF_FOCUS_PQ,
        "designs": summaries,
        "exact_validation_improved_over_clear": validation_improved,
        "all_support_checks_passed": support_passed,
        "passed": bool(validation_improved and support_passed),
    }
    (args.output_dir / "matched-designs.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Gate 8 failed; do not open locked test metrics")


if __name__ == "__main__":
    main()
