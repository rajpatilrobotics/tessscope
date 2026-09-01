"""Run hard InstanSeg validation for all promoted v2 designs before test freeze."""

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
from tessscope.v2.autofocus.features import spectral_features
from tessscope.v2.autofocus.metrics import focus_metrics
from tessscope.v2.autofocus.ridge import ridge_predict
from tessscope.v2.optics.model import simulate_cubic_rate, simulate_noisy_sensor
from tessscope.v2.optimization.served import (
    DEPTHS,
    V2SystemCalibration,
    collect_patches,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "optimization" / "summary.json"
)
CUBIC = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "optimization"
    / "cubic-baseline.json"
)
OUTPUT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "gate4" / "hard-validation.json"
)


def design_registry() -> dict[str, dict]:
    optimization = json.loads(OPTIMIZATION.read_text())
    designs = optimization["designs"]
    selected_name = designs["selected_exact_joint"]
    selected_joint = next(
        row for row in designs["exact_joint_starts"] if row["name"] == selected_name
    )
    cubic = json.loads(CUBIC.read_text())["selected"]
    return {
        "clear": {"family": "zernike", "parameters": [0.0] * 6},
        "cubic": {
            "family": "cubic",
            "strength": cubic["rms_strength_radians"],
        },
        "astigmatic": {
            "family": "zernike",
            "parameters": designs["classical_astigmatic"]["final_parameters"],
        },
        "segmentation_only": {
            "family": "zernike",
            "parameters": designs["segmentation_only"]["final_parameters"],
        },
        "focus_only": {
            "family": "zernike",
            "parameters": designs["focus_only"]["final_parameters"],
        },
        "exact_joint": {
            "family": "zernike",
            "parameters": selected_joint["final_parameters"],
            "source": selected_name,
        },
        "broken_focus_gradient": {
            "family": "zernike",
            "parameters": designs["broken_focus_gradient"]["final_parameters"],
        },
        "naive_superposition": {
            "family": "zernike",
            "parameters": designs["naive_superposition"]["final_parameters"],
        },
        "derivative_free": {
            "family": "zernike",
            "parameters": designs["derivative_free"]["final_parameters"],
        },
    }


def simulate_design(
    design: dict,
    objects: np.ndarray,
    depths: np.ndarray,
    calibration: V2SystemCalibration,
) -> np.ndarray:
    if design["family"] == "cubic":
        sensor = simulate_cubic_rate(
            jnp.asarray(design["strength"], dtype=jnp.float32),
            jnp.asarray(objects),
            jnp.asarray(depths),
            exposure_gain=calibration.exposure_gain,
            depth_scale=calibration.depth_scale,
            axial_offset_um=calibration.axial_offset_um,
        )
    else:
        noise = np.zeros((len(objects), len(depths), 256, 256), dtype=np.float32)
        sensor = simulate_noisy_sensor(
            jnp.asarray(design["parameters"], dtype=jnp.float32),
            jnp.asarray(objects),
            jnp.asarray(depths),
            jnp.asarray(noise),
            expected_photons=calibration.expected_photons,
            exposure_gain=calibration.exposure_gain,
            depth_scale=calibration.depth_scale,
            axial_offset_um=calibration.axial_offset_um,
        )
    return np.maximum(np.asarray(sensor), 0.0)


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Hard-validation artifact already exists: {OUTPUT}")
    calibration = V2SystemCalibration.load()
    patches = collect_patches("validation", 12, seed=53)
    instance_counts = np.asarray(
        [np.count_nonzero(patch.valid_objects) for patch in patches]
    )
    hard_threshold = int(np.ceil(np.median(instance_counts)))
    hard_fields = {
        patch.field_id
        for patch, count in zip(patches, instance_counts, strict=True)
        if count >= hard_threshold
    }
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = ObserverTransform(
        mode="affine", offset=0.0, scale=calibration.exposure_gain
    )
    designs = design_registry()
    summaries = {}
    all_rows = []
    corrected_rows = []
    for design_name, design in designs.items():
        design_rows = []
        target_depths = []
        predicted_depths = []
        photon_means = []
        for query_index, query in enumerate(patches):
            support = [
                patches[(query_index + 1) % len(patches)],
                patches[(query_index + 2) % len(patches)],
            ]
            objects = np.stack(
                [support[0].object_image, support[1].object_image, query.object_image]
            ).astype(np.float32)
            sensor = simulate_design(design, objects, DEPTHS, calibration)
            photon_means.append(float(np.mean(sensor) * calibration.expected_photons))
            features, _ = spectral_features(sensor.reshape(-1, 256, 256))
            ridge = ridge_predict(
                features[:14],
                np.tile(DEPTHS, 2),
                features[14:],
                ridge_lambda=calibration.ridge_lambda,
            )
            target_depths.extend(DEPTHS)
            predicted_depths.extend(ridge.predictions)
            predictions = segment_sensor_batch(
                model,
                sensor[2],
                query.instance_labels.shape,
                device=device,
                maximum_batch_size=4,
                transform=transform,
            )
            for depth, predicted_depth, prediction in zip(
                DEPTHS, ridge.predictions, predictions, strict=True
            ):
                target, valid_prediction = valid_region_labels(
                    query.instance_labels, prediction, margin=62
                )
                metrics = instance_metrics(target, valid_prediction)
                row = {
                    "design": design_name,
                    "well": query.well,
                    "field_id": query.field_id,
                    "depth_um": float(depth),
                    "predicted_depth_um": float(predicted_depth),
                    "stage_action_um": float(np.clip(-predicted_depth, -6.0, 6.0)),
                    "hard_dense_patch": query.field_id in hard_fields,
                    "valid_instance_count": int(
                        np.count_nonzero(query.valid_objects)
                    ),
                    **metrics.to_dict(),
                }
                design_rows.append(row)
                all_rows.append(row)

            if design_name == "exact_joint":
                residual_depths = DEPTHS + np.clip(-ridge.predictions, -6.0, 6.0)
                corrected_sensor = simulate_design(
                    design,
                    query.object_image[None].astype(np.float32),
                    residual_depths.astype(np.float32),
                    calibration,
                )[0]
                corrected_predictions = segment_sensor_batch(
                    model,
                    corrected_sensor,
                    query.instance_labels.shape,
                    device=device,
                    maximum_batch_size=4,
                    transform=transform,
                )
                for original, residual, prediction in zip(
                    design_rows[-7:], residual_depths, corrected_predictions, strict=True
                ):
                    target, valid_prediction = valid_region_labels(
                        query.instance_labels, prediction, margin=62
                    )
                    corrected_rows.append(
                        {
                            "well": query.well,
                            "field_id": query.field_id,
                            "original_depth_um": original["depth_um"],
                            "residual_depth_um": float(residual),
                            "hard_dense_patch": original["hard_dense_patch"],
                            "before_pq": original["panoptic_quality"],
                            "after_pq": instance_metrics(
                                target, valid_prediction
                            ).panoptic_quality,
                        }
                    )
        off_focus = [row for row in design_rows if row["depth_um"] != 0]
        hard = [row for row in off_focus if row["hard_dense_patch"]]
        by_depth = {
            str(float(depth)): float(
                np.mean(
                    [
                        row["panoptic_quality"]
                        for row in design_rows
                        if row["depth_um"] == float(depth)
                    ]
                )
            )
            for depth in DEPTHS
        }
        summaries[design_name] = {
            "mean_off_focus_pq": float(
                np.mean([row["panoptic_quality"] for row in off_focus])
            ),
            "hard_dense_off_focus_pq": float(
                np.mean([row["panoptic_quality"] for row in hard])
            ),
            "focus_pq": by_depth["0.0"],
            "worst_off_focus_depth_pq": min(
                value for depth, value in by_depth.items() if depth != "0.0"
            ),
            "pq_by_depth": by_depth,
            "mean_photon_count": float(np.mean(photon_means)),
            **focus_metrics(
                np.asarray(target_depths), np.asarray(predicted_depths)
            ),
        }
        print(json.dumps({"design": design_name, **summaries[design_name]}), flush=True)

    joint = summaries["exact_joint"]
    clear = summaries["clear"]
    segmentation = summaries["segmentation_only"]
    superposed = summaries["naive_superposition"]
    derivative_free = summaries["derivative_free"]
    hard_corrected = [
        row
        for row in corrected_rows
        if row["hard_dense_patch"] and row["original_depth_um"] != 0
    ]
    stage = {
        "mean_absolute_depth_before_um": float(
            np.mean([abs(row["original_depth_um"]) for row in corrected_rows])
        ),
        "mean_absolute_depth_after_um": float(
            np.mean([abs(row["residual_depth_um"]) for row in corrected_rows])
        ),
        "hard_off_focus_pq_before": float(
            np.mean([row["before_pq"] for row in hard_corrected])
        ),
        "hard_off_focus_pq_after": float(
            np.mean([row["after_pq"] for row in hard_corrected])
        ),
    }
    gates = {
        "joint_hard_pq_gain_over_clear": (
            joint["hard_dense_off_focus_pq"] - clear["hard_dense_off_focus_pq"]
        ),
        "joint_hard_pq_drop_from_segmentation_only": (
            segmentation["hard_dense_off_focus_pq"]
            - joint["hard_dense_off_focus_pq"]
        ),
        "joint_pareto_dominates_superposition": bool(
            joint["hard_dense_off_focus_pq"]
            > superposed["hard_dense_off_focus_pq"]
            and joint["mae_um"] < superposed["mae_um"]
        ),
        "joint_pareto_dominates_derivative_free": bool(
            joint["hard_dense_off_focus_pq"]
            > derivative_free["hard_dense_off_focus_pq"]
            and joint["mae_um"] < derivative_free["mae_um"]
        ),
        "stage_reduces_absolute_defocus": bool(
            stage["mean_absolute_depth_after_um"]
            < stage["mean_absolute_depth_before_um"]
        ),
        "stage_improves_hard_pq": bool(
            stage["hard_off_focus_pq_after"] > stage["hard_off_focus_pq_before"]
        ),
    }
    gates["passed"] = bool(
        gates["joint_hard_pq_gain_over_clear"] >= 0.01
        and gates["joint_hard_pq_drop_from_segmentation_only"] <= 0.01
        and gates["joint_pareto_dominates_superposition"]
        and gates["joint_pareto_dominates_derivative_free"]
        and gates["stage_reduces_absolute_defocus"]
        and gates["stage_improves_hard_pq"]
    )
    report = {
        "status": "complete_validation_only",
        "test_accessed": False,
        "hard_definition": {
            "rule": "denser half of query patches by target valid-instance count",
            "threshold_instances": hard_threshold,
            "hard_field_ids": sorted(hard_fields),
            "primary_depths_um": [-6, -4, -2, 2, 4, 6],
        },
        "design_registry": designs,
        "summaries": summaries,
        "stage_correction": stage,
        "gates": gates,
        "rows": all_rows,
        "corrected_rows": corrected_rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "stage": stage, "gates": gates}, indent=2))
    if not gates["passed"]:
        raise SystemExit("Hard validation gates failed; test access remains blocked")


if __name__ == "__main__":
    main()
