"""Hard-validate the three post-gate Pareto refinements without test access."""

from __future__ import annotations

import argparse
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
from tessscope.v2.optics.model import simulate_noisy_sensor
from tessscope.v2.optimization.served import (
    DEPTHS,
    V2SystemCalibration,
    collect_patches,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFINEMENT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "optimization"
    / "pareto-refinement.json"
)
FIRST_VALIDATION = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "gate4" / "hard-validation.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "gate4"
    / "hard-validation-refinement.json"
)


def simulate(
    parameters: np.ndarray,
    objects: np.ndarray,
    depths: np.ndarray,
    calibration: V2SystemCalibration,
) -> np.ndarray:
    noise = np.zeros((len(objects), len(depths), 256, 256), dtype=np.float32)
    return np.maximum(
        np.asarray(
            simulate_noisy_sensor(
                jnp.asarray(parameters),
                jnp.asarray(objects),
                jnp.asarray(depths),
                jnp.asarray(noise),
                expected_photons=calibration.expected_photons,
                exposure_gain=calibration.exposure_gain,
                depth_scale=calibration.depth_scale,
                axial_offset_um=calibration.axial_offset_um,
            )
        ),
        0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refinement", type=Path, default=REFINEMENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Pareto hard-validation already exists: {args.output}")
    first = json.loads(FIRST_VALIDATION.read_text())
    refinement = json.loads(args.refinement.read_text())
    references = first["summaries"]
    hard_fields = set(first["hard_definition"]["hard_field_ids"])
    calibration = V2SystemCalibration.load()
    patches = collect_patches("validation", 12, seed=53)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = ObserverTransform(
        mode="affine", offset=0.0, scale=calibration.exposure_gain
    )
    summaries = {}
    all_rows = []
    corrected_rows = []
    for candidate in refinement["candidates"]:
        name = candidate["name"]
        parameters = np.asarray(candidate["final_parameters"], dtype=np.float32)
        rows = []
        target_depths = []
        predicted_depths = []
        for query_index, query in enumerate(patches):
            support = [
                patches[(query_index + 1) % len(patches)],
                patches[(query_index + 2) % len(patches)],
            ]
            objects = np.stack(
                [support[0].object_image, support[1].object_image, query.object_image]
            ).astype(np.float32)
            sensor = simulate(parameters, objects, DEPTHS, calibration)
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
                    "candidate": name,
                    "well": query.well,
                    "field_id": query.field_id,
                    "depth_um": float(depth),
                    "predicted_depth_um": float(predicted_depth),
                    "hard_dense_patch": query.field_id in hard_fields,
                    **instance_metrics(target, valid_prediction).to_dict(),
                }
                rows.append(row)
                query_rows.append(row)
                all_rows.append(row)

            residual_depths = DEPTHS + np.clip(-ridge.predictions, -6.0, 6.0)
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
                target, valid_prediction = valid_region_labels(
                    query.instance_labels, prediction, margin=62
                )
                corrected_rows.append(
                    {
                        "candidate": name,
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
        off_focus = [row for row in rows if row["depth_um"] != 0]
        hard = [row for row in off_focus if row["hard_dense_patch"]]
        corrected = [
            row
            for row in corrected_rows
            if row["candidate"] == name
            and row["hard_dense_patch"]
            and row["original_depth_um"] != 0
        ]
        summary = {
            "focus_weight": candidate.get("focus_weight"),
            "gradient_balance": candidate.get("gradient_balance"),
            "mean_off_focus_pq": float(
                np.mean([row["panoptic_quality"] for row in off_focus])
            ),
            "hard_dense_off_focus_pq": float(
                np.mean([row["panoptic_quality"] for row in hard])
            ),
            **focus_metrics(
                np.asarray(target_depths), np.asarray(predicted_depths)
            ),
            "mean_absolute_depth_after_correction_um": float(
                np.mean([abs(row["residual_depth_um"]) for row in corrected])
            ),
            "hard_pq_before_correction": float(
                np.mean([row["before_pq"] for row in corrected])
            ),
            "hard_pq_after_correction": float(
                np.mean([row["after_pq"] for row in corrected])
            ),
        }
        summary["gates"] = {
            "gain_over_clear": (
                summary["hard_dense_off_focus_pq"]
                - references["clear"]["hard_dense_off_focus_pq"]
            ),
            "drop_from_segmentation_only": (
                references["segmentation_only"]["hard_dense_off_focus_pq"]
                - summary["hard_dense_off_focus_pq"]
            ),
            "pareto_superposition": bool(
                summary["hard_dense_off_focus_pq"]
                > references["naive_superposition"]["hard_dense_off_focus_pq"]
                and summary["mae_um"] < references["naive_superposition"]["mae_um"]
            ),
            "pareto_derivative_free": bool(
                summary["hard_dense_off_focus_pq"]
                > references["derivative_free"]["hard_dense_off_focus_pq"]
                and summary["mae_um"] < references["derivative_free"]["mae_um"]
            ),
        }
        summary["gates"]["passed"] = bool(
            summary["gates"]["gain_over_clear"] >= 0.01
            and summary["gates"]["drop_from_segmentation_only"] <= 0.01
            and summary["gates"]["pareto_superposition"]
            and summary["gates"]["pareto_derivative_free"]
            and summary["signed_direction_accuracy"] >= 0.90
            and summary["mae_um"] <= 2.0
            and summary["mean_absolute_depth_after_correction_um"]
            < summary["mean_absolute_depth_before_correction_um"]
            and summary["hard_pq_after_correction"]
            > summary["hard_pq_before_correction"]
        )
        summaries[name] = summary
        print(json.dumps({"candidate": name, **summary}), flush=True)

    eligible = [name for name, row in summaries.items() if row["gates"]["passed"]]
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
        "status": "complete_validation_only",
        "test_accessed": False,
        "reason": refinement.get("reason", refinement.get("method", "validation refinement")),
        "summaries": summaries,
        "eligible_candidates": eligible,
        "selected_candidate": selected,
        "passed": selected is not None,
        "rows": all_rows,
        "corrected_rows": corrected_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": selected,
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    if selected is None:
        raise SystemExit("No Pareto refinement passed hard validation; test remains blocked")


if __name__ == "__main__":
    main()
