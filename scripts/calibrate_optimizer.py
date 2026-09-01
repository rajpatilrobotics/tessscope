"""Freeze the surrogate scale and Adam learning rate before full design runs."""

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
from tesseract_jax import apply_tesseract

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
from tessscope.optics.model import phase_coefficients
from tessscope.optimization.adam import AdamState, adam_update
from tessscope.optimization.schedule import build_training_schedule
from tessscope.optimization.training import (
    materialize_batch,
    observer_sensor_gradient_norms,
    task_value_and_gradient,
)

LEARNING_RATES = (0.01, 0.03, 0.1)
PILOT_STEPS = 24
SURROGATE_CALIBRATION_BATCHES = 32
DEPTHS_UM = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
PRIMARY_DEPTHS_UM = {-6.0, -4.0, -2.0, 2.0, 4.0, 6.0}


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
        "--output",
        type=Path,
        default=Path("artifacts/runs/gate7/optimizer-calibration.json"),
    )
    return parser.parse_args()


def calibrate_surrogate_scale(
    optics: Tesseract,
    observer: Tesseract,
) -> tuple[float, list[dict[str, float | int]]]:
    schedule = build_training_schedule(steps=SURROGATE_CALIBRATION_BATCHES)
    parameters = jnp.zeros((6,), dtype=jnp.float32)
    rows: list[dict[str, float | int]] = []
    for spec in schedule:
        batch = materialize_batch(spec)
        formed = apply_tesseract(
            optics,
            {
                "phase_parameters": parameters,
                "object_batch": batch.object_batch,
                "depths_um": batch.depths_um,
                "expected_photons": 100.0,
            },
        )
        exact_norm, proxy_norm = observer_sensor_gradient_norms(
            observer,
            np.asarray(formed["sensor"]),
            np.asarray(formed["phase_coefficients"]),
            batch,
        )
        ratio = exact_norm / (proxy_norm + 1e-12)
        rows.append(
            {
                "step": spec.step,
                "exact_sensor_gradient_norm": exact_norm,
                "proxy_sensor_gradient_norm": proxy_norm,
                "norm_ratio": ratio,
            }
        )
    scale = float(np.median([row["norm_ratio"] for row in rows]))
    return scale, rows


def run_pilot(
    optics: Tesseract,
    observer: Tesseract,
    learning_rate: float,
) -> tuple[np.ndarray, list[dict[str, object]], float]:
    parameters = np.zeros((6,), dtype=np.float32)
    state = AdamState.zeros(parameters.shape)
    trace = []
    started = time.perf_counter()
    for spec in build_training_schedule(steps=PILOT_STEPS):
        batch = materialize_batch(spec)
        value, gradient = task_value_and_gradient(
            optics, observer, parameters, batch, backward_mode="exact"
        )
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError(f"Non-finite exact pilot at step {spec.step}")
        parameters, state = adam_update(
            parameters,
            gradient,
            state,
            learning_rate=learning_rate,
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
    return parameters, trace, time.perf_counter() - started


def validation_score(parameters: np.ndarray) -> ValidationScore:
    normalization = ObjectNormalization(125.0, 1642.0)
    patches = [
        prepare_patch(record, origin, normalization)
        for record in records_for_split("validation")[:10]
        for origin in patch_origins("validation")
    ]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = load_frozen_transform()
    margin = int(
        np.floor(48 * BIOLOGICAL_SAMPLING_UM / OBSERVER_SAMPLING_UM + 0.5)
    )
    rows: list[tuple[float, float]] = []
    for patch in patches:
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


def main() -> None:
    args = parse_args()
    optics = Tesseract.from_url(args.optics_url, timeout=120)
    observer = Tesseract.from_url(args.observer_url, timeout=120)
    optics.health()
    observer.health()

    surrogate_scale, surrogate_rows = calibrate_surrogate_scale(optics, observer)
    pilots = []
    for learning_rate in LEARNING_RATES:
        parameters, trace, wall_seconds = run_pilot(optics, observer, learning_rate)
        score = validation_score(parameters)
        pilots.append(
            {
                "learning_rate": learning_rate,
                "parameters": parameters.tolist(),
                "phase_coefficients": np.asarray(
                    phase_coefficients(jnp.asarray(parameters))
                ).tolist(),
                "phase_rms_radians": float(
                    np.linalg.norm(np.asarray(phase_coefficients(jnp.asarray(parameters))))
                ),
                "validation": asdict(score),
                "wall_seconds": wall_seconds,
                "trace": trace,
            }
        )
    selected = max(
        pilots,
        key=lambda pilot: (
            pilot["validation"]["mean_off_focus_pq"],
            pilot["validation"]["worst_depth_pq"],
            -pilot["learning_rate"],
        ),
    )
    report = {
        "gate": "gate7_optimizer_and_surrogate_calibration",
        "training_only_surrogate_calibration": {
            "batch_count": SURROGATE_CALIBRATION_BATCHES,
            "rule": "median(||g_exact||_2 / (||h_proxy||_2 + 1e-12))",
            "frozen_scale": surrogate_scale,
            "rows": surrogate_rows,
        },
        "adam": {
            "candidate_learning_rates": list(LEARNING_RATES),
            "pilot_steps": PILOT_STEPS,
            "validation_source_count": 10,
            "selection_rule": (
                "mean off-focus PQ, then worst-depth PQ, then lower learning rate"
            ),
            "pilots": pilots,
            "selected_learning_rate": selected["learning_rate"],
        },
        "passed": bool(np.isfinite(surrogate_scale) and surrogate_scale > 0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Gate 7 optimizer calibration failed")


if __name__ == "__main__":
    main()
