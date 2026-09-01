"""Run the frozen 45-well hard endpoint for all selected v2.2 mixtures."""

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
from tessscope.v2.optimization.served import (
    DEPTHS,
    V2SystemCalibration,
    collect_patches,
)
from tessscope.v2_1.evaluation import paired_hard_pq_bootstrap
from tessscope.v2_1.optics.model import simulate_noisy_sensor_b7
from tessscope.v2_2.evaluation import (
    hard_pareto_names,
    hypervolume_2d,
    paired_focus_mae_bootstrap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREEN = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "screening"
    / "piecewise-soft-frontier.json"
)
V2_1_HARD = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "validation"
    / "expanded-hard-validation.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "validation"
    / "piecewise-hard-frontier.json"
)
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "artifacts" / "runtime-runs" / "v2_2-hard-frontier"
)
JOINT_NAME = "b7-projected-0.50-step-30"
HARD_PQ_EFFECT = 0.005
FOCUS_MAE_EFFECT_UM = 0.10


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


def summarize(rows: list[dict], photon_means: list[float]) -> dict:
    off_focus = [row for row in rows if row["depth_um"] != 0.0]
    hard = [row for row in off_focus if row["hard_dense_patch"]]
    targets = np.asarray([row["depth_um"] for row in rows])
    predictions = np.asarray([row["predicted_depth_um"] for row in rows])
    summary = {
        "mean_photon_count": float(np.mean(photon_means)),
        "sample_count": len(rows),
        "hard_off_focus_sample_count": len(hard),
        **focus_metrics(targets, predictions),
    }
    for metric, label in (
        ("panoptic_quality", "pq"),
        ("recognition_quality", "rq"),
        ("segmentation_quality", "sq"),
        ("foreground_dice", "dice"),
        ("absolute_count_error", "absolute_count_error"),
        ("percentage_count_error", "percentage_count_error"),
    ):
        summary[f"hard_dense_off_focus_{label}"] = float(
            np.mean([row[metric] for row in hard])
        )
    return summary


def evaluate_design(
    point: dict,
    patches,
    hard_fields: set[str],
    calibration: V2SystemCalibration,
    model,
    device: torch.device,
    transform: ObserverTransform,
) -> dict:
    parameters = np.asarray(point["parameters"], dtype=np.float32)
    rows = []
    corrected_rows = []
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
                "design": point["name"],
                "well": query.well,
                "field_id": query.field_id,
                "depth_um": float(depth),
                "predicted_depth_um": float(predicted_depth),
                "stage_action_um": float(np.clip(-predicted_depth, -6.0, 6.0)),
                "hard_dense_patch": query.field_id in hard_fields,
                "valid_instance_count": int(np.count_nonzero(query.valid_objects)),
                **instance_metrics(target, valid_prediction).to_dict(),
            }
            rows.append(row)
            query_rows.append(row)
        residual_depths = DEPTHS + np.clip(-ridge.predictions, -6.0, 6.0)
        if query.field_id not in hard_fields:
            continue
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
            query_rows, residual_depths, corrected_predictions, strict=True
        ):
            if original["depth_um"] == 0.0:
                continue
            target, valid_prediction = valid_region_labels(
                query.instance_labels, prediction, margin=62
            )
            corrected_rows.append(
                {
                    "design": point["name"],
                    "well": query.well,
                    "field_id": query.field_id,
                    "original_depth_um": original["depth_um"],
                    "residual_depth_um": float(residual),
                    "before_pq": original["panoptic_quality"],
                    "after_pq": instance_metrics(
                        target, valid_prediction
                    ).panoptic_quality,
                }
            )
    design_summary = summarize(rows, photon_means)
    hard_primary = [
        row for row in rows if row["hard_dense_patch"] and row["depth_um"] != 0.0
    ]
    stage = {
        "mean_absolute_depth_before_um": float(
            np.mean([abs(row["depth_um"]) for row in rows])
        ),
        "mean_absolute_depth_after_um": float(
            np.mean(
                [
                    abs(row["depth_um"] + row["stage_action_um"])
                    for row in rows
                ]
            )
        ),
        "hard_off_focus_pq_before": float(
            np.mean([row["panoptic_quality"] for row in hard_primary])
        ),
        "hard_off_focus_pq_after": float(
            np.mean([row["after_pq"] for row in corrected_rows])
        ),
        "fraction_hard_frames_improved": float(
            np.mean([row["after_pq"] > row["before_pq"] for row in corrected_rows])
        ),
    }
    return {
        "name": point["name"],
        "summary": design_summary,
        "stage_correction": stage,
        "rows": rows,
        "corrected_rows": corrected_rows,
    }


def load_or_evaluate(
    point: dict,
    patches,
    hard_fields,
    calibration,
    model,
    device,
    transform,
) -> dict:
    checkpoint = CHECKPOINT_ROOT / f"{point['name']}.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())
    result = evaluate_design(
        point,
        patches,
        hard_fields,
        calibration,
        model,
        device,
        transform,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(result) + "\n")
    return result


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"V2.2 hard-frontier artifact already exists: {OUTPUT}")
    screen = json.loads(SCREEN.read_text())
    old = json.loads(V2_1_HARD.read_text())
    selected = set(screen["selected_for_hard_validation"])
    points = [point for point in screen["points"] if point["name"] in selected]
    if len(points) != screen["selected_point_count"]:
        raise SystemExit("Could not resolve all frozen v2.2 hard-frontier points")
    patches = collect_patches("validation", 45, seed=53)
    if [patch.field_id for patch in patches] != sorted(
        [
            row["field_id"]
            for row in old["rows"]
            if row["design"] == "clear" and row["depth_um"] == 0.0
        ],
        key=lambda field_id: next(
            index for index, patch in enumerate(patches) if patch.field_id == field_id
        ),
    ):
        raise SystemExit("V2.2 patch order differs from the frozen v2.1 protocol")
    hard_fields = set(old["hard_definition"]["hard_field_ids"])
    calibration = V2SystemCalibration.load()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = ObserverTransform(
        mode="affine", offset=0.0, scale=calibration.exposure_gain
    )
    results = []
    for index, point in enumerate(points, start=1):
        result = load_or_evaluate(
            point,
            patches,
            hard_fields,
            calibration,
            model,
            device,
            transform,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(points),
                    "design": point["name"],
                    "hard_pq": result["summary"]["hard_dense_off_focus_pq"],
                    "mae_um": result["summary"]["mae_um"],
                    "corrected_pq": result["stage_correction"][
                        "hard_off_focus_pq_after"
                    ],
                }
            ),
            flush=True,
        )
    summaries = {result["name"]: result["summary"] for result in results}
    stages = {result["name"]: result["stage_correction"] for result in results}
    all_rows = [row for result in results for row in result["rows"]]
    all_corrected = [
        row for result in results for row in result["corrected_rows"]
    ]
    joint_rows = [row for row in old["rows"] if row["design"] == JOINT_NAME]
    comparison_rows = [*joint_rows, *all_rows]
    joint = old["summaries"][JOINT_NAME]
    pareto = hard_pareto_names(summaries)
    matched_segmentation = [
        name
        for name, row in summaries.items()
        if row["hard_dense_off_focus_pq"]
        >= joint["hard_dense_off_focus_pq"]
    ]
    best_at_joint_segmentation = (
        min(matched_segmentation, key=lambda name: (summaries[name]["mae_um"], name))
        if matched_segmentation
        else None
    )
    matched_focus = [
        name for name, row in summaries.items() if row["mae_um"] <= joint["mae_um"]
    ]
    best_at_joint_focus = (
        max(
            matched_focus,
            key=lambda name: (
                summaries[name]["hard_dense_off_focus_pq"],
                -summaries[name]["mae_um"],
            ),
        )
        if matched_focus
        else None
    )
    segmentation_evidence = None
    if best_at_joint_segmentation is not None:
        segmentation_evidence = paired_focus_mae_bootstrap(
            comparison_rows, JOINT_NAME, best_at_joint_segmentation
        )
        segmentation_evidence["piecewise_name"] = best_at_joint_segmentation
        segmentation_evidence["passed"] = bool(
            segmentation_evidence["mean_difference"] >= FOCUS_MAE_EFFECT_UM
            and segmentation_evidence["ci_lower_95"] > 0.0
        )
    focus_evidence = None
    if best_at_joint_focus is not None:
        focus_evidence = paired_hard_pq_bootstrap(
            comparison_rows, JOINT_NAME, best_at_joint_focus
        )
        focus_evidence["piecewise_name"] = best_at_joint_focus
        focus_evidence["passed"] = bool(
            focus_evidence["mean_difference"] >= HARD_PQ_EFFECT
            and focus_evidence["ci_lower_95"] > 0.0
        )
    clear = old["summaries"]["clear"]
    segmentation = old["summaries"]["b7_segmentation_only"]
    naive = old["summaries"]["b7_naive_superposition"]
    pq_scale = segmentation["hard_dense_off_focus_pq"] - clear["hard_dense_off_focus_pq"]
    focus_scale = segmentation["mae_um"] - naive["mae_um"]

    def normalized(row: dict) -> tuple[float, float]:
        return (
            (row["hard_dense_off_focus_pq"] - clear["hard_dense_off_focus_pq"])
            / pq_scale,
            (segmentation["mae_um"] - row["mae_um"]) / focus_scale,
        )

    frontier_hypervolume = hypervolume_2d(normalized(summaries[name]) for name in pareto)
    with_joint_hypervolume = hypervolume_2d(
        [*(normalized(summaries[name]) for name in pareto), normalized(joint)]
    )
    frontier_passed = bool(
        (segmentation_evidence and segmentation_evidence["passed"])
        or (focus_evidence and focus_evidence["passed"])
    )
    report = {
        "status": (
            "passed_matched_frontier_validation_only"
            if frontier_passed
            else "failed_matched_frontier_validation_only"
        ),
        "test_accessed": False,
        "frozen_screen_path": str(SCREEN.relative_to(PROJECT_ROOT)),
        "v2_1_reference_path": str(V2_1_HARD.relative_to(PROJECT_ROOT)),
        "coverage": old["coverage"],
        "hard_definition": old["hard_definition"],
        "selected_point_count": len(points),
        "summaries": summaries,
        "stage_correction": stages,
        "hard_pareto_names": sorted(pareto),
        "matched_segmentation_evidence": segmentation_evidence,
        "matched_focus_evidence": focus_evidence,
        "hypervolume": {
            "reference": [0.0, 0.0],
            "piecewise_frontier": frontier_hypervolume,
            "piecewise_plus_joint": with_joint_hypervolume,
            "joint_increment": with_joint_hypervolume - frontier_hypervolume,
        },
        "application_evidence": {
            JOINT_NAME: {
                "first_frame_hard_pq": joint["hard_dense_off_focus_pq"],
                "predicted_stage_mean_residual_um": old["stage_correction"][JOINT_NAME][
                    "mean_absolute_depth_after_um"
                ],
                "second_frame_hard_pq": old["stage_correction"][JOINT_NAME][
                    "hard_off_focus_pq_after"
                ],
            }
        },
        "frontier_passed": frontier_passed,
        "next_gate": (
            "matched_b7_gradient_free_control"
            if frontier_passed
            else "preserve_v2_2_and_start_v2_3"
        ),
        "rows": all_rows,
        "corrected_rows": all_corrected,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "frontier_passed": frontier_passed,
                "best_at_joint_segmentation": best_at_joint_segmentation,
                "best_at_joint_focus": best_at_joint_focus,
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
