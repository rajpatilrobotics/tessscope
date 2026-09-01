"""Select a classical astigmatic focus baseline on train/validation simulated stacks."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tessscope.v2.autofocus.features import spectral_features
from tessscope.v2.autofocus.metrics import focus_metrics
from tessscope.v2.autofocus.ridge import ridge_predict
from tessscope.v2.data.bbbc006 import records_for_split
from tessscope.v2.data.patches import patch_origins, prepare_patch
from tessscope.v2.data.preprocess import (
    accepted_registration_fields,
    load_global_normalization,
)
from tessscope.v2.optics.model import astigmatic_parameters, simulate_noisy_sensor

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
OUTPUT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "gate2" / "astigmatic-autofocus.json"
)
DEPTHS = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
STRENGTHS = (0.5, 1.0, 1.5, 2.0, 2.4)


def select_objects(split: str, count: int) -> np.ndarray:
    normalization = load_global_normalization()
    accepted = accepted_registration_fields()
    objects = []
    for record in records_for_split(split):
        if record.field_id not in accepted:
            continue
        patch = prepare_patch(record, patch_origins(split)[0], normalization)
        if np.count_nonzero(patch.valid_objects) == 0:
            continue
        objects.append(patch.object_image)
        if len(objects) == count:
            return np.stack(objects).astype(np.float32)
    raise ValueError(f"Could not collect {count} accepted {split} object patches")


def design_features(
    parameters: np.ndarray,
    objects: np.ndarray,
    *,
    exposure_gain: float,
    depth_scale: float,
    axial_offset_um: float,
) -> np.ndarray:
    rows = []
    for start in range(0, len(objects), 4):
        batch = objects[start : start + 4]
        noise = np.zeros((len(batch), 7, 256, 256), dtype=np.float32)
        sensor = np.asarray(
            simulate_noisy_sensor(
                jnp.asarray(parameters),
                jnp.asarray(batch),
                jnp.asarray(DEPTHS),
                jnp.asarray(noise),
                expected_photons=200.0,
                exposure_gain=exposure_gain,
                depth_scale=depth_scale,
                axial_offset_um=axial_offset_um,
            )
        )
        features, _ = spectral_features(sensor.reshape(-1, 256, 256))
        rows.append(features)
    return np.concatenate(rows)


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text())
    training_calibration = json.loads(TRAINING_CALIBRATION.read_text())
    exposure_gain = float(training_calibration["exposure"]["gain"])
    depth_scale = float(calibration["model"]["fitted_depth_scale"])
    axial_offset_um = float(calibration["model"]["fitted_axial_offset_um"])
    training_objects = select_objects("training", 24)
    validation_objects = select_objects("validation", 12)
    training_depths = np.tile(DEPTHS, len(training_objects))
    validation_depths = np.tile(DEPTHS, len(validation_objects))
    candidates = []
    for orientation in (0, 1):
        for strength in STRENGTHS:
            parameters = astigmatic_parameters(strength, orientation=orientation)
            training_features = design_features(
                parameters,
                training_objects,
                exposure_gain=exposure_gain,
                depth_scale=depth_scale,
                axial_offset_um=axial_offset_um,
            )
            validation_features = design_features(
                parameters,
                validation_objects,
                exposure_gain=exposure_gain,
                depth_scale=depth_scale,
                axial_offset_um=axial_offset_um,
            )
            ridge = ridge_predict(
                training_features,
                training_depths,
                validation_features,
                ridge_lambda=100.0,
            )
            candidates.append(
                {
                    "orientation": orientation,
                    "rms_strength_radians": strength,
                    "phase_parameters": parameters.tolist(),
                    **focus_metrics(validation_depths, ridge.predictions),
                }
            )
            print(
                json.dumps(
                    {
                        "orientation": orientation,
                        "strength": strength,
                        "mae_um": candidates[-1]["mae_um"],
                    }
                ),
                flush=True,
            )
    selected = min(candidates, key=lambda row: (row["mae_um"], -row["signed_direction_accuracy"]))
    passed = bool(
        selected["signed_direction_accuracy"] >= 0.90 and selected["mae_um"] <= 2.0
    )
    report = {
        "design_family": "classical_single_mode_astigmatism",
        "fit_split": "24 deterministic training patches",
        "selection_split": "12 deterministic validation patches",
        "test_accessed": False,
        "ridge_lambda": 100.0,
        "calibration": {
            "depth_scale": depth_scale,
            "axial_offset_um": axial_offset_um,
            "exposure_gain": exposure_gain,
        },
        "candidates": candidates,
        "selected": selected,
        "thresholds": {"signed_direction_accuracy": 0.90, "mae_um": 2.0},
        "passed": passed,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "selected": selected, "passed": passed}, indent=2))
    if not passed:
        raise SystemExit("Classical astigmatic autofocus feasibility gate failed")


if __name__ == "__main__":
    main()
