"""Gate the full served B7 optics→autofocus→InstanSeg derivative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from tessscope.v2.data.bbbc006 import records_for_split
from tessscope.v2.data.patches import patch_origins, prepare_patch
from tessscope.v2.data.preprocess import (
    accepted_registration_fields,
    load_global_normalization,
)
from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import DEPTHS, V2SystemCalibration
from tessscope.validation.derivatives import directional_derivative_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2_1" / "gates" / "b7-three-tesseract-derivative.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--basis-size", type=int, choices=(7, 11), default=7)
    return parser.parse_args()


def derivative_patches():
    """Select two deterministic training-only patches for the derivative gate."""
    normalization = load_global_normalization()
    accepted = accepted_registration_fields()
    selected = []
    for record in records_for_split("training"):
        if record.field_id not in accepted:
            continue
        for origin in patch_origins("training"):
            patch = prepare_patch(record, origin, normalization)
            if np.count_nonzero(patch.valid_objects) > 0:
                selected.append(patch)
            if len(selected) == 2:
                return selected
    raise ValueError("Could not find two accepted training patches with valid instances")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"B7 derivative artifact already exists: {args.output}")
    calibration = V2SystemCalibration.load()
    patches = derivative_patches()
    optics = Tesseract.from_url(args.optics_url, timeout=180)
    autofocus = Tesseract.from_url(args.autofocus_url, timeout=180)
    observer = Tesseract.from_url(args.observer_url, timeout=180)
    for service in (optics, autofocus, observer):
        service.health()

    objects = np.stack([patch.object_image for patch in patches]).astype(np.float32)
    optics_static = {
        "object_batch": objects,
        "depths_um": DEPTHS,
        "noise_standard_normal": np.zeros((2, 7, 256, 256), dtype=np.float32),
        "expected_photons": calibration.expected_photons,
        "exposure_gain": calibration.exposure_gain,
        "depth_scale": calibration.depth_scale,
        "axial_offset_um": calibration.axial_offset_um,
    }
    autofocus_static = {
        "support_depth_um": DEPTHS,
        "query_depth_um": DEPTHS,
        "ridge_lambda": calibration.ridge_lambda,
        "radial_bins": 10,
        "angular_bins": 12,
    }
    observer_static = {
        "instance_labels": patches[1].instance_labels[None].astype(np.int32),
        "centers_yx": patches[1].centers_yx[None].astype(np.float32),
        "valid_objects": patches[1].valid_objects[None].astype(np.uint8),
        "transform_mode": "affine",
        "transform_offset": 0.0,
        "transform_scale": calibration.exposure_gain,
        "backward_mode": "exact",
        "surrogate_scale": 1.0,
    }
    weights = JointObjectiveWeights()

    def branches(parameters: jax.Array) -> tuple[jax.Array, jax.Array]:
        formed = apply_tesseract(
            optics,
            {"phase_parameters": parameters, **optics_static},
        )
        focused = apply_tesseract(
            autofocus,
            {
                "support_sensor": formed["sensor"][0],
                "query_sensor": formed["sensor"][1],
                **autofocus_static,
            },
        )
        segmented = apply_tesseract(
            observer,
            {
                "sensor": formed["sensor"][1:2],
                "phase_coefficients": formed["phase_coefficients"],
                **observer_static,
            },
        )
        return segmented["task_loss"], focused["focus_mse"]

    def objective(parameters: jax.Array) -> jax.Array:
        segmentation_loss, focus_mse = branches(parameters)
        return weights.combine(segmentation_loss, focus_mse)

    point = np.linspace(-0.02, 0.04, args.basis_size, dtype=np.float32)
    value, gradient = jax.value_and_grad(objective)(jnp.asarray(point))
    branch_values = branches(jnp.asarray(point))
    segmentation_gradient = jax.grad(lambda candidate: branches(candidate)[0])(jnp.asarray(point))
    focus_gradient = jax.grad(lambda candidate: branches(candidate)[1])(jnp.asarray(point))
    report = directional_derivative_report(
        lambda candidate: float(objective(jnp.asarray(candidate, dtype=jnp.float32))),
        point,
        np.asarray(gradient),
        seed=71,
        minimum_stable_epsilons=2,
    )
    report.update(
        {
            "component": (
                f"v2_1_b{args.basis_size}_full_served_"
                "jax_scipy_pytorch_chain"
            ),
            "objective_value": float(value),
            "point": point.tolist(),
            "basis_noll_indices": list(range(5, 5 + args.basis_size)),
            "patch_ids": [
                f"{patch.field_id}:{patch.origin_yx[0]}:{patch.origin_yx[1]}" for patch in patches
            ],
            "branch_values": {
                "segmentation_loss": float(branch_values[0]),
                "focus_mse": float(branch_values[1]),
            },
            "branch_gradients": {
                "segmentation_loss": np.asarray(segmentation_gradient).tolist(),
                "focus_mse": np.asarray(focus_gradient).tolist(),
            },
            "ridge_lambda": calibration.ridge_lambda,
            "test_accessed": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "relative_error": report["overall_median_relative_error"],
                "cosine": report["overall_cosine_agreement"],
                "passed": report["passed"],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(
            f"B{args.basis_size} full three-Tesseract derivative gate failed"
        )


if __name__ == "__main__":
    main()
