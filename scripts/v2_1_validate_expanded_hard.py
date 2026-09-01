"""Run the maximum valid well-grouped hard screen for frozen B7 candidates."""

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
from tessscope.v2.data.bbbc006 import records_for_split
from tessscope.v2.data.patches import patch_origins, prepare_patch
from tessscope.v2.data.preprocess import (
    accepted_registration_fields,
    load_global_normalization,
)
from tessscope.v2.optimization.served import (
    DEPTHS,
    V2SystemCalibration,
    collect_patches,
)
from tessscope.v2_1.evaluation import paired_hard_pq_bootstrap
from tessscope.v2_1.optics.model import (
    simulate_noisy_sensor_b7,
    unconstrained_from_coefficients_b7,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_SUMMARY = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "optimization" / "summary.json"
)
OPTIMIZATION_ROOT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2_1" / "optimization"
)
SEPARATE = OPTIMIZATION_ROOT / "b7-separate-baselines.json"
PROJECTED = OPTIMIZATION_ROOT / "b7-projected-gradient-candidates.json"
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "validation"
    / "expanded-hard-validation.json"
)
TARGET_VALIDATION_WELLS = 48
EVALUATED_VALIDATION_WELLS = 45


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def selected_projected_rows(projected: dict) -> list[dict]:
    selected = set(projected["selected_for_expanded_hard_validation"])
    rows = [
        row
        for family in projected["candidates"]
        for row in family["full_training_contenders"]
        if row["name"] in selected
    ]
    if len(rows) != len(selected):
        raise ValueError("Could not resolve every projected hard-validation candidate")
    return sorted(rows, key=lambda row: row["name"])


def design_registry() -> dict[str, dict]:
    separate = load_json(SEPARATE)
    projected = load_json(PROJECTED)
    v2_designs = load_json(V2_SUMMARY)["designs"]
    derivative_free_coefficients = np.asarray(
        [*v2_designs["derivative_free"]["final_phase_coefficients"], 0.0],
        dtype=np.float64,
    )
    registry = {
        "clear": {
            "parameters": np.zeros(7, dtype=np.float32).tolist(),
            "role": "clear B7 zero pupil",
        },
        "b7_segmentation_only": {
            "parameters": separate["segmentation_only"]["final_parameters"],
            "role": "matched 90-step B7 segmentation-only baseline",
        },
        "b7_naive_superposition": {
            "parameters": separate["naive_superposition"]["final_parameters"],
            "role": "matched B7 separate-pupil naive superposition",
        },
        "legacy_derivative_free_extended_b7": {
            "parameters": unconstrained_from_coefficients_b7(
                derivative_free_coefficients
            ).tolist(),
            "role": (
                "v2 matched derivative-free B6 control extended with zero Noll 11; "
                "not a matched B7 derivative-free search"
            ),
        },
    }
    for row in selected_projected_rows(projected):
        registry[row["name"]] = {
            "parameters": row["parameters"],
            "role": "frozen projected exact-gradient B7 candidate",
        }
    return registry


def simulate(
    parameters: np.ndarray,
    objects: np.ndarray,
    depths: np.ndarray,
    calibration: V2SystemCalibration,
) -> np.ndarray:
    noise = np.zeros((len(objects), len(depths), 256, 256), dtype=np.float32)
    return np.maximum(
        np.asarray(
            simulate_noisy_sensor_b7(
                jnp.asarray(parameters, dtype=jnp.float32),
                jnp.asarray(objects, dtype=jnp.float32),
                jnp.asarray(depths, dtype=jnp.float32),
                jnp.asarray(noise),
                expected_photons=calibration.expected_photons,
                exposure_gain=calibration.exposure_gain,
                depth_scale=calibration.depth_scale,
                axial_offset_um=calibration.axial_offset_um,
            )
        ),
        0.0,
    )


def coverage_exclusions(patches) -> list[dict]:
    """Explain every frozen validation well excluded by existing quality rules."""
    records = records_for_split("validation")
    all_wells = {record.well for record in records}
    selected_wells = {patch.well for patch in patches}
    accepted = accepted_registration_fields()
    normalization = load_global_normalization()
    exclusions = []
    for well in sorted(all_wells - selected_wells):
        well_records = [record for record in records if record.well == well]
        accepted_records = [
            record for record in well_records if record.field_id in accepted
        ]
        valid_patch_count = 0
        for record in accepted_records:
            for origin in patch_origins("validation"):
                patch = prepare_patch(record, origin, normalization)
                valid_patch_count += int(np.count_nonzero(patch.valid_objects) > 0)
        reason = (
            "no field passed the frozen registration gate"
            if not accepted_records
            else "registered fields contained no valid supervised patch instances"
        )
        exclusions.append(
            {
                "well": well,
                "field_ids": [record.field_id for record in well_records],
                "accepted_registration_field_ids": [
                    record.field_id for record in accepted_records
                ],
                "valid_patch_count": valid_patch_count,
                "reason": reason,
            }
        )
    return exclusions


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Expanded hard-validation artifact already exists: {OUTPUT}")
    calibration = V2SystemCalibration.load()
    patches = collect_patches(
        "validation", EVALUATED_VALIDATION_WELLS, seed=53
    )
    if len({patch.well for patch in patches}) != EVALUATED_VALIDATION_WELLS:
        raise SystemExit("Expanded validation patches must have distinct wells")
    exclusions = coverage_exclusions(patches)
    if len(patches) + len(exclusions) != TARGET_VALIDATION_WELLS:
        raise SystemExit("Expanded validation coverage audit does not total 48 wells")
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
    candidate_names = [
        name
        for name, design in designs.items()
        if design["role"] == "frozen projected exact-gradient B7 candidate"
    ]
    summaries = {}
    all_rows = []
    corrected_rows = []
    for design_name, design in designs.items():
        parameters = np.asarray(design["parameters"], dtype=np.float32)
        rows = []
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
            sensor = simulate(parameters, objects, DEPTHS, calibration)
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
            query_rows = []
            for depth, predicted_depth, prediction in zip(
                DEPTHS, ridge.predictions, predictions, strict=True
            ):
                target, valid_prediction = valid_region_labels(
                    query.instance_labels, prediction, margin=62
                )
                row = {
                    "design": design_name,
                    "well": query.well,
                    "field_id": query.field_id,
                    "depth_um": float(depth),
                    "predicted_depth_um": float(predicted_depth),
                    "stage_action_um": float(
                        np.clip(-predicted_depth, -6.0, 6.0)
                    ),
                    "hard_dense_patch": query.field_id in hard_fields,
                    "valid_instance_count": int(
                        np.count_nonzero(query.valid_objects)
                    ),
                    **instance_metrics(target, valid_prediction).to_dict(),
                }
                rows.append(row)
                query_rows.append(row)
                all_rows.append(row)
            if design_name in candidate_names:
                residual_depths = DEPTHS + np.clip(
                    -ridge.predictions, -6.0, 6.0
                )
                corrected_sensor = simulate(
                    parameters,
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
                    query_rows,
                    residual_depths,
                    corrected_predictions,
                    strict=True,
                ):
                    target, valid_prediction = valid_region_labels(
                        query.instance_labels, prediction, margin=62
                    )
                    corrected_rows.append(
                        {
                            "design": design_name,
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
        off_focus = [row for row in rows if row["depth_um"] != 0.0]
        hard = [row for row in off_focus if row["hard_dense_patch"]]
        by_depth = {
            str(float(depth)): float(
                np.mean(
                    [
                        row["panoptic_quality"]
                        for row in rows
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
        print(
            json.dumps({"design": design_name, **summaries[design_name]}),
            flush=True,
        )
    bootstraps = {
        candidate_name: {
            reference_name: paired_hard_pq_bootstrap(
                all_rows, candidate_name, reference_name
            )
            for reference_name in (
                "clear",
                "b7_segmentation_only",
                "b7_naive_superposition",
                "legacy_derivative_free_extended_b7",
            )
        }
        for candidate_name in candidate_names
    }
    stages = {}
    gates = {}
    for candidate_name in candidate_names:
        corrected = [
            row
            for row in corrected_rows
            if row["design"] == candidate_name
        ]
        hard_corrected = [
            row
            for row in corrected
            if row["hard_dense_patch"] and row["original_depth_um"] != 0.0
        ]
        stages[candidate_name] = {
            "mean_absolute_depth_before_um": float(
                np.mean([abs(row["original_depth_um"]) for row in corrected])
            ),
            "mean_absolute_depth_after_um": float(
                np.mean([abs(row["residual_depth_um"]) for row in corrected])
            ),
            "hard_off_focus_pq_before": float(
                np.mean([row["before_pq"] for row in hard_corrected])
            ),
            "hard_off_focus_pq_after": float(
                np.mean([row["after_pq"] for row in hard_corrected])
            ),
        }
        candidate = summaries[candidate_name]
        clear = summaries["clear"]
        segmentation = summaries["b7_segmentation_only"]
        superposed = summaries["b7_naive_superposition"]
        derivative_free = summaries["legacy_derivative_free_extended_b7"]
        stage = stages[candidate_name]
        clear_bootstrap = bootstraps[candidate_name]["clear"]
        candidate_gates = {
            "hard_pq_gain_over_clear": (
                candidate["hard_dense_off_focus_pq"]
                - clear["hard_dense_off_focus_pq"]
            ),
            "clear_gain_bootstrap_ci_lower_positive": bool(
                clear_bootstrap["ci_lower_95"] > 0.0
            ),
            "hard_pq_drop_from_segmentation_only": (
                segmentation["hard_dense_off_focus_pq"]
                - candidate["hard_dense_off_focus_pq"]
            ),
            "pareto_dominates_naive_superposition": bool(
                candidate["hard_dense_off_focus_pq"]
                > superposed["hard_dense_off_focus_pq"]
                and candidate["mae_um"] < superposed["mae_um"]
            ),
            "pareto_dominates_legacy_derivative_free_control": bool(
                candidate["hard_dense_off_focus_pq"]
                > derivative_free["hard_dense_off_focus_pq"]
                and candidate["mae_um"] < derivative_free["mae_um"]
            ),
            "signed_direction_accuracy_at_least_0.90": bool(
                candidate["signed_direction_accuracy"] >= 0.90
            ),
            "focus_mae_at_most_2_um": bool(candidate["mae_um"] <= 2.0),
            "stage_reduces_absolute_defocus": bool(
                stage["mean_absolute_depth_after_um"]
                < stage["mean_absolute_depth_before_um"]
            ),
            "stage_improves_hard_pq": bool(
                stage["hard_off_focus_pq_after"]
                > stage["hard_off_focus_pq_before"]
            ),
            "positive_mean_photon_count": bool(
                candidate["mean_photon_count"] > 0.0
            ),
        }
        candidate_gates["passed_preliminary_without_matched_b7_derivative_free"] = bool(
            candidate_gates["hard_pq_gain_over_clear"] >= 0.01
            and candidate_gates["clear_gain_bootstrap_ci_lower_positive"]
            and candidate_gates["hard_pq_drop_from_segmentation_only"] <= 0.01
            and candidate_gates["pareto_dominates_naive_superposition"]
            and candidate_gates[
                "pareto_dominates_legacy_derivative_free_control"
            ]
            and candidate_gates["signed_direction_accuracy_at_least_0.90"]
            and candidate_gates["focus_mae_at_most_2_um"]
            and candidate_gates["stage_reduces_absolute_defocus"]
            and candidate_gates["stage_improves_hard_pq"]
            and candidate_gates["positive_mean_photon_count"]
        )
        gates[candidate_name] = candidate_gates
    eligible = [
        name
        for name, candidate_gates in gates.items()
        if candidate_gates["passed_preliminary_without_matched_b7_derivative_free"]
    ]
    selected = (
        max(
            eligible,
            key=lambda name: (
                summaries[name]["hard_dense_off_focus_pq"],
                -summaries[name]["mae_um"],
            ),
        )
        if eligible
        else None
    )
    report = {
        "status": "complete_expanded_45_of_48_well_validation_only",
        "test_accessed": False,
        "coverage": {
            "target_validation_wells": TARGET_VALIDATION_WELLS,
            "evaluated_validation_wells": len(patches),
            "excluded_validation_wells": len(exclusions),
            "exclusions": exclusions,
        },
        "hard_definition": {
            "well_count": len(patches),
            "one_deterministic_field_per_well": True,
            "rule": "denser half by target valid-instance count",
            "threshold_instances": hard_threshold,
            "hard_field_ids": sorted(hard_fields),
            "primary_depths_um": [-6, -4, -2, 2, 4, 6],
            "bootstrap_replicates": 2000,
            "bootstrap_unit": "well",
        },
        "design_registry": designs,
        "summaries": summaries,
        "paired_well_bootstraps": bootstraps,
        "stage_correction": stages,
        "gates": gates,
        "eligible_for_matched_b7_derivative_free": eligible,
        "selected_for_matched_b7_derivative_free": selected,
        "passed_preliminary": selected is not None,
        "rows": all_rows,
        "corrected_rows": corrected_rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selected_for_matched_b7_derivative_free": selected,
                "passed_preliminary": report["passed_preliminary"],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if selected is None:
        raise SystemExit(
            "No projected B7 candidate passed expanded hard validation; "
            "test access remains blocked"
        )


if __name__ == "__main__":
    main()
