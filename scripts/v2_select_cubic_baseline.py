"""Select the matched cubic baseline using validation hard PQ and signed focus."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from tessscope.evaluation.metrics import instance_metrics, valid_region_labels
from tessscope.evaluation.route import segment_sensor_batch
from tessscope.observer.instanseg import load_frozen_model
from tessscope.observer.loss import ObserverTransform
from tessscope.optics.model import cubic_support_energy_fraction
from tessscope.v2.autofocus.features import spectral_features
from tessscope.v2.autofocus.metrics import focus_metrics
from tessscope.v2.autofocus.ridge import ridge_predict
from tessscope.v2.optics.model import simulate_cubic_rate
from tessscope.v2.optimization.served import (
    DEPTHS,
    V2SystemCalibration,
    collect_patches,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "optimization" / "cubic-baseline.json"
)
STRENGTHS = (0.0, 0.6, 1.2, 1.8, 2.4)


def main() -> None:
    calibration = V2SystemCalibration.load()
    patches = collect_patches("validation", 12, seed=53)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = ObserverTransform(
        mode="affine", offset=0.0, scale=calibration.exposure_gain
    )
    margin = 62
    candidates = []
    for strength in STRENGTHS:
        pq_rows = []
        predicted_depths = []
        target_depths = []
        photon_means = []
        for start in range(0, len(patches), 3):
            batch = patches[start : start + 3]
            objects = np.stack([patch.object_image for patch in batch]).astype(np.float32)
            sensor = np.maximum(
                np.asarray(
                    simulate_cubic_rate(
                        jnp.asarray(strength, dtype=jnp.float32),
                        jnp.asarray(objects),
                        jnp.asarray(DEPTHS),
                        exposure_gain=calibration.exposure_gain,
                        depth_scale=calibration.depth_scale,
                        axial_offset_um=calibration.axial_offset_um,
                    )
                ),
                0.0,
            )
            photon_means.append(float(np.mean(sensor) * calibration.expected_photons))
            features, _ = spectral_features(sensor.reshape(-1, 256, 256))
            ridge = ridge_predict(
                features[:14],
                np.tile(DEPTHS, 2),
                features[14:],
                ridge_lambda=calibration.ridge_lambda,
            )
            predicted_depths.extend(ridge.predictions)
            target_depths.extend(DEPTHS)
            predictions = segment_sensor_batch(
                model,
                sensor[2],
                batch[2].instance_labels.shape,
                device=device,
                maximum_batch_size=4,
                transform=transform,
            )
            for depth, prediction in zip(DEPTHS, predictions, strict=True):
                target, valid_prediction = valid_region_labels(
                    batch[2].instance_labels, prediction, margin
                )
                pq_rows.append(
                    {
                        "depth_um": float(depth),
                        "panoptic_quality": instance_metrics(
                            target, valid_prediction
                        ).panoptic_quality,
                    }
                )
        off_focus = [
            row["panoptic_quality"] for row in pq_rows if row["depth_um"] != 0
        ]
        focus = [
            row["panoptic_quality"] for row in pq_rows if row["depth_um"] == 0
        ]
        candidate = {
            "rms_strength_radians": strength,
            "mean_off_focus_pq": float(np.mean(off_focus)),
            "worst_depth_pq": float(
                min(
                    np.mean(
                        [
                            row["panoptic_quality"]
                            for row in pq_rows
                            if row["depth_um"] == float(depth)
                        ]
                    )
                    for depth in DEPTHS
                    if depth != 0
                )
            ),
            "focus_pq": float(np.mean(focus)),
            "mean_photon_count": float(np.mean(photon_means)),
            **focus_metrics(
                np.asarray(target_depths), np.asarray(predicted_depths)
            ),
        }
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
    clear = candidates[0]
    for candidate in candidates:
        candidate["normalized_validation_utility"] = (
            (1.0 - candidate["mean_off_focus_pq"])
            / max(1.0 - clear["mean_off_focus_pq"], 1e-6)
            + candidate["mae_um"] / max(clear["mae_um"], 1e-6)
        )
    selected = min(
        candidates,
        key=lambda row: (row["normalized_validation_utility"], -row["worst_depth_pq"]),
    )
    support = {
        str(float(depth)): float(
            cubic_support_energy_fraction(
                jnp.asarray(selected["rms_strength_radians"], dtype=jnp.float32),
                jnp.asarray(depth * calibration.depth_scale + calibration.axial_offset_um),
            )
        )
        for depth in DEPTHS
    }
    report = {
        "design_family": "RMS-normalized cubic x^3+y^3 pupil",
        "split": "validation wells only",
        "test_accessed": False,
        "patch_count": len(patches),
        "selection_rule": "minimum normalized segmentation-error plus signed-focus-MAE utility",
        "candidates": candidates,
        "selected": selected,
        "support_energy_fraction_by_depth": support,
        "minimum_support_energy_fraction": min(support.values()),
        "minimum_required_support_energy_fraction": 0.995,
        "passed": bool(min(support.values()) >= 0.995),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(OUTPUT), "selected": selected, "passed": report["passed"]},
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("Selected cubic pupil exceeds the frozen PSF support")


if __name__ == "__main__":
    main()
