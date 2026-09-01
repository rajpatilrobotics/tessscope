"""Run the frozen five-direction gate through both HTTP Tesseracts."""

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
from tessscope.validation.derivatives import directional_derivative_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8401")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument(
        "--output", default="artifacts/runs/gate5/served-directional-derivative.json"
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

    optics_static = {
        "object_batch": patch.object_image[None].astype(np.float32),
        "depths_um": np.asarray([-6.0, 0.0, 6.0], dtype=np.float32),
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

    def jax_objective(parameters: jax.Array) -> jax.Array:
        formed = apply_tesseract(
            optics, {"phase_parameters": parameters, **optics_static}
        )
        observed = apply_tesseract(
            observer,
            {
                "sensor": formed["sensor"],
                "phase_coefficients": formed["phase_coefficients"],
                **observer_static,
            },
        )
        return observed["task_loss"]

    # A small nonzero interior point avoids exact clear-pupil symmetries while
    # remaining representative of the first optimization updates.
    point = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03], dtype=np.float32)
    value, gradient = jax.value_and_grad(jax_objective)(jnp.asarray(point))

    def scalar_objective(parameters: np.ndarray) -> float:
        return float(jax_objective(jnp.asarray(parameters, dtype=jnp.float32)))

    report = directional_derivative_report(
        scalar_objective, point, np.asarray(gradient), seed=29
    )
    report.update(
        {
            "component": "full_served_optics_to_observer_chain",
            "source_image_id": patch.image_id,
            "point": point.tolist(),
            "objective_value": float(value),
            "center_gradient_policy": (
                "Connected predicted center embeddings, explicitly approved as the "
                "2026-09-01 option-1 contract amendment."
            ),
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Served derivative gate failed")


if __name__ == "__main__":
    main()
