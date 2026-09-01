"""Check the served optics and observer derivative components separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from tessscope.data.bbbc039 import (
    ObjectNormalization,
    patch_origins,
    prepare_patch,
    records_for_split,
)
from tessscope.optics.model import phase_coefficients, simulate_sensor
from tessscope.validation.derivatives import directional_derivative_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8401")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument(
        "--output", default="artifacts/runs/gate5/component-derivatives.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch = prepare_patch(
        records_for_split("training")[0],
        patch_origins("training")[0],
        ObjectNormalization(125.0, 1642.0),
    )
    optics = Tesseract.from_url(args.optics_url, timeout=120)
    observer = Tesseract.from_url(args.observer_url, timeout=120)
    optics.health()
    observer.health()

    object_batch = patch.object_image[None].astype(np.float32)
    depths = np.asarray([-6.0, 0.0, 6.0], dtype=np.float32)
    optics_static = {
        "object_batch": object_batch,
        "depths_um": depths,
        "expected_photons": 100.0,
    }
    observer_static = {
        "instance_labels": patch.instance_labels[None].astype(np.int32),
        "centers_yx": patch.centers_yx[None].astype(np.float32),
        "valid_objects": patch.valid_objects[None].astype(np.uint8),
        "transform_mode": "identity",
        "transform_offset": 0.0,
        "transform_scale": 1.0,
    }
    base_parameters = np.asarray(
        [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03], dtype=np.float32
    )

    def optics_objective(parameters: jax.Array) -> jax.Array:
        formed = apply_tesseract(
            optics, {"phase_parameters": parameters, **optics_static}
        )
        sensor_term = 10.0 * jnp.mean(jnp.square(formed["sensor"]))
        phase_term = 0.1 * jnp.mean(jnp.square(formed["phase_coefficients"]))
        return sensor_term + phase_term

    optics_value, optics_gradient = jax.value_and_grad(optics_objective)(
        jnp.asarray(base_parameters)
    )
    optics_report = directional_derivative_report(
        lambda value: float(optics_objective(jnp.asarray(value, dtype=jnp.float32))),
        base_parameters,
        np.asarray(optics_gradient),
        seed=29,
    )
    optics_report.update(
        {
            "component": "served_jax_chromatix_optics",
            "objective_value": float(optics_value),
            "point": base_parameters.tolist(),
        }
    )

    local_objects = jnp.asarray(object_batch)
    local_depths = jnp.asarray(depths)

    def local_formed(parameters: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            simulate_sensor(parameters, local_objects, local_depths),
            phase_coefficients(parameters),
        )

    baseline_sensor, baseline_coefficients = local_formed(
        jnp.asarray(base_parameters)
    )
    basis = jnp.eye(6, dtype=jnp.float32)
    tangent_pairs = [
        jax.jvp(local_formed, (jnp.asarray(base_parameters),), (direction,))[1]
        for direction in basis
    ]
    sensor_tangents = jnp.stack([pair[0] for pair in tangent_pairs])
    coefficient_tangents = jnp.stack([pair[1] for pair in tangent_pairs])

    def observer_path_objective(path_parameters: jax.Array) -> jax.Array:
        sensor = baseline_sensor + jnp.tensordot(
            path_parameters, sensor_tangents, axes=1
        )
        coefficients = baseline_coefficients + jnp.tensordot(
            path_parameters, coefficient_tangents, axes=1
        )
        return apply_tesseract(
            observer,
            {
                "sensor": sensor,
                "phase_coefficients": coefficients,
                **observer_static,
            },
        )["task_loss"]

    observer_point = np.zeros(6, dtype=np.float32)
    observer_value, observer_gradient = jax.value_and_grad(observer_path_objective)(
        jnp.asarray(observer_point)
    )
    observer_report = directional_derivative_report(
        lambda value: float(
            observer_path_objective(jnp.asarray(value, dtype=jnp.float32))
        ),
        observer_point,
        np.asarray(observer_gradient),
        seed=29,
    )
    observer_report.update(
        {
            "component": "served_pytorch_instanseg_observer",
            "objective_value": float(observer_value),
            "path": "linearized local-optics tangent basis at the gate point",
        }
    )

    report = {
        "source_image_id": patch.image_id,
        "optics": optics_report,
        "observer": observer_report,
        "passed": bool(optics_report["passed"] and observer_report["passed"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("A served component derivative gate failed")


if __name__ == "__main__":
    main()
