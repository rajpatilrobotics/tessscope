"""Select and freeze the global observer transform using validation crops only."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from tessscope.data.bbbc039 import (
    BIOLOGICAL_SAMPLING_UM,
    OBSERVER_SAMPLING_UM,
    ObjectNormalization,
    patch_origins,
    prepare_patch,
    records_for_split,
)
from tessscope.evaluation.metrics import instance_metrics, valid_region_labels
from tessscope.evaluation.route import deterministic_sensor, segment_sensor_batch
from tessscope.observer.calibration import (
    TransformScore,
    UnitIntervalHistogram,
    choose_transform,
    transform_candidates,
)
from tessscope.observer.instanseg import load_frozen_model, normalize_release_input

DEPTHS_UM = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
PRIMARY_DEPTHS_UM = {-6.0, -4.0, -2.0, 2.0, 4.0, 6.0}
PHASE_PARAMETERS = np.zeros((6,), dtype=np.float32)
VALID_MARGIN_BIOLOGICAL_PX = 48


def split_patches(split: str):
    normalization = ObjectNormalization(lower=125.0, upper=1642.0)
    for record in records_for_split(split):
        for origin in patch_origins(split):
            yield prepare_patch(record, origin, normalization)


def calibrated_bounds() -> tuple[float, float, int]:
    histogram = UnitIntervalHistogram()
    patch_count = 0
    for patch in split_patches("training"):
        sensor = deterministic_sensor(PHASE_PARAMETERS, patch.object_image, DEPTHS_UM)
        histogram.update(sensor)
        patch_count += 1
    return histogram.percentile(0.1), histogram.percentile(99.9), patch_count


def standard_labels(
    model: torch.nn.Module,
    sensor: np.ndarray,
    observer_shape: tuple[int, int],
    device: torch.device,
    maximum_batch_size: int = 4,
) -> np.ndarray:
    labels = []
    with torch.inference_mode():
        for start in range(0, len(sensor), maximum_batch_size):
            chunk = torch.from_numpy(sensor[start : start + maximum_batch_size].copy())
            resized = functional.interpolate(
                chunk[:, None], size=observer_shape, mode="bilinear", align_corners=False
            )
            normalized = normalize_release_input(resized.numpy())
            labels.append(
                model(torch.from_numpy(normalized).to(device))[:, 0]
                .to("cpu")
                .numpy()
                .astype(np.int32)
            )
    return np.concatenate(labels)


def main() -> None:
    lower, upper, patch_count = calibrated_bounds()
    candidates = transform_candidates(lower, upper)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    observer_margin = int(
        np.floor(
            VALID_MARGIN_BIOLOGICAL_PX
            * BIOLOGICAL_SAMPLING_UM
            / OBSERVER_SAMPLING_UM
            + 0.5
        )
    )
    per_candidate: dict[str, list[dict[str, float | str | list[int]]]] = {
        name: [] for name in candidates
    }
    standard_focus_rows: list[float] = []

    for patch in split_patches("validation"):
        sensor = deterministic_sensor(PHASE_PARAMETERS, patch.object_image, DEPTHS_UM)
        focus_index = int(np.flatnonzero(DEPTHS_UM == 0)[0])
        standard_prediction = standard_labels(
            model,
            sensor[focus_index : focus_index + 1],
            patch.instance_labels.shape,
            device,
        )[0]
        target, prediction = valid_region_labels(
            patch.instance_labels, standard_prediction, observer_margin
        )
        standard_focus_rows.append(instance_metrics(target, prediction).panoptic_quality)
        for name, transform in candidates.items():
            predictions = segment_sensor_batch(
                model,
                sensor,
                patch.instance_labels.shape,
                device=device,
                maximum_batch_size=4,
                transform=transform,
            )
            for depth, prediction in zip(DEPTHS_UM, predictions, strict=True):
                target, prediction = valid_region_labels(
                    patch.instance_labels, prediction, observer_margin
                )
                metrics = instance_metrics(target, prediction)
                per_candidate[name].append(
                    {
                        "source_image_id": patch.image_id,
                        "origin_yx": list(patch.origin_yx),
                        "depth_um": float(depth),
                        "panoptic_quality": metrics.panoptic_quality,
                        "absolute_count_error": metrics.absolute_count_error,
                    }
                )

    scores: list[TransformScore] = []
    depth_summaries: dict[str, dict[str, float]] = {}
    for name, rows in per_candidate.items():
        by_depth = {
            float(depth): float(
                np.mean(
                    [
                        row["panoptic_quality"]
                        for row in rows
                        if row["depth_um"] == float(depth)
                    ]
                )
            )
            for depth in DEPTHS_UM
        }
        primary = [by_depth[depth] for depth in sorted(PRIMARY_DEPTHS_UM)]
        score = TransformScore(
            name=name,
            mean_off_focus_pq=float(np.mean(primary)),
            worst_depth_pq=float(min(primary)),
            focus_pq=by_depth[0.0],
        )
        scores.append(score)
        depth_summaries[name] = {str(depth): value for depth, value in by_depth.items()}

    standard_focus_pq = float(np.mean(standard_focus_rows))
    maximum_focus_drop = 0.03
    selected = choose_transform(
        scores,
        standard_focus_pq=standard_focus_pq,
        maximum_focus_drop=maximum_focus_drop,
    )
    selected_transform = candidates[selected.name]
    report = {
        "gate": "gate7_observer_transform_selection",
        "split": "validation",
        "host": platform.platform(),
        "device": str(device),
        "training_calibration_patch_count": patch_count,
        "validation_patch_count": len(standard_focus_rows),
        "depths_um": DEPTHS_UM.tolist(),
        "evaluation_margin_observer_px": observer_margin,
        "calibration": {
            "histogram_bins": 65_536,
            "lower_percentile": 0.1,
            "lower_value": lower,
            "upper_percentile": 99.9,
            "upper_value": upper,
        },
        "scores": [score.to_dict() for score in scores],
        "mean_pq_by_depth": depth_summaries,
        "standard_per_image_focus_pq": standard_focus_pq,
        "maximum_allowed_focus_pq_drop": maximum_focus_drop,
        "minimum_acceptable_global_focus_pq": standard_focus_pq
        - maximum_focus_drop,
        "selection_rule": (
            "use global affine if it passes the focus-PQ gate; otherwise require "
            "global asinh to pass"
        ),
        "selected": {
            "name": selected.name,
            "mode": selected_transform.mode,
            "offset": selected_transform.offset,
            "scale": selected_transform.scale,
        },
        "passed": True,
    }
    output = Path("artifacts/runs/gate7/observer-transform-selection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
