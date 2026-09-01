"""Run the frozen full derivative gate across optics, autofocus, and InstanSeg."""

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
from tessscope.validation.derivatives import directional_derivative_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "gate3"
    / "clear-calibration-offset.json"
)
TRAINING_CALIBRATION = (
    PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-training-calibration.json"
)
DEPTHS = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument(
        "--output",
        default="artifacts/runs/v2/gate3/three-tesseract-derivative-contract.json",
    )
    return parser.parse_args()


def derivative_patches():
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
    calibration = json.loads(CALIBRATION.read_text())
    if not calibration["passed"]:
        raise SystemExit("Clear digital-twin calibration gate has not passed")
    training_calibration = json.loads(TRAINING_CALIBRATION.read_text())
    exposure_gain = float(training_calibration["exposure"]["gain"])
    depth_scale = float(calibration["model"]["fitted_depth_scale"])
    axial_offset_um = float(calibration["model"]["fitted_axial_offset_um"])
    patches = derivative_patches()

    optics = Tesseract.from_url(args.optics_url, timeout=180)
    autofocus = Tesseract.from_url(args.autofocus_url, timeout=180)
    observer = Tesseract.from_url(args.observer_url, timeout=180)
    for service in (optics, autofocus, observer):
        service.health()

    objects = np.stack([patch.object_image for patch in patches]).astype(np.float32)
    labels = patches[1].instance_labels[None].astype(np.int32)
    centers = patches[1].centers_yx[None].astype(np.float32)
    valid = patches[1].valid_objects[None].astype(np.uint8)
    optics_static = {
        "object_batch": objects,
        "depths_um": DEPTHS,
        "noise_standard_normal": np.zeros((2, 7, 256, 256), dtype=np.float32),
        "expected_photons": 200.0,
        "exposure_gain": exposure_gain,
        "depth_scale": depth_scale,
        "axial_offset_um": axial_offset_um,
    }
    autofocus_static = {
        "support_depth_um": DEPTHS,
        "query_depth_um": DEPTHS,
        "ridge_lambda": 1.0,
        "radial_bins": 10,
        "angular_bins": 12,
    }
    observer_static = {
        "instance_labels": labels,
        "centers_yx": centers,
        "valid_objects": valid,
        "transform_mode": "affine",
        "transform_offset": 0.0,
        "transform_scale": exposure_gain,
        "backward_mode": "exact",
        "surrogate_scale": 1.0,
    }
    weights = JointObjectiveWeights()

    def branches(parameters: jax.Array) -> tuple[jax.Array, jax.Array]:
        formed = apply_tesseract(
            optics, {"phase_parameters": parameters, **optics_static}
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

    point = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03], dtype=np.float32)
    value, gradient = jax.value_and_grad(objective)(jnp.asarray(point))
    branch_values = branches(jnp.asarray(point))
    segmentation_gradient = jax.grad(lambda value: branches(value)[0])(
        jnp.asarray(point)
    )
    focus_gradient = jax.grad(lambda value: branches(value)[1])(jnp.asarray(point))
    report = directional_derivative_report(
        lambda candidate: float(objective(jnp.asarray(candidate, dtype=jnp.float32))),
        point,
        np.asarray(gradient),
        seed=43,
        minimum_stable_epsilons=2,
    )
    report.update(
        {
            "component": "full_served_jax_scipy_pytorch_three_tesseract_chain",
            "objective_value": float(value),
            "point": point.tolist(),
            "patch_ids": [
                f"{patch.field_id}:{patch.origin_yx[0]}:{patch.origin_yx[1]}"
                for patch in patches
            ],
            "branch_values": {
                "segmentation_loss": float(branch_values[0]),
                "focus_mse": float(branch_values[1]),
            },
            "branch_gradients": {
                "segmentation_loss": np.asarray(segmentation_gradient).tolist(),
                "focus_mse": np.asarray(focus_gradient).tolist(),
            },
            "weights": {
                "segmentation": weights.segmentation,
                "focus": weights.focus,
                "focus_normalizer_um2": weights.maximum_depth_um**2,
            },
            "depth_scale": depth_scale,
            "axial_offset_um": axial_offset_um,
            "exposure_gain": exposure_gain,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "overall_median_relative_error": report[
                    "overall_median_relative_error"
                ],
                "overall_cosine_agreement": report["overall_cosine_agreement"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("Full served three-Tesseract derivative gate failed")


if __name__ == "__main__":
    main()
