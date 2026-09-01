"""Instance-segmentation metrics with source-image grouped bootstrap intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class InstanceMetrics:
    """One image/depth's locked instance-segmentation measurements."""

    panoptic_quality: float
    segmentation_quality: float
    recognition_quality: float
    true_positives: int
    false_positives: int
    false_negatives: int
    predicted_count: int
    target_count: int
    absolute_count_error: int
    percentage_count_error: float
    foreground_dice: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _canonical_positive_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    value = np.asarray(labels)
    positive = np.unique(value[value > 0])
    remapped = np.zeros(value.shape, dtype=np.int32)
    if positive.size:
        indices = np.searchsorted(positive, value[value > 0]) + 1
        remapped[value > 0] = indices.astype(np.int32)
    return remapped, int(positive.size)


def _intersection_over_union(
    target: np.ndarray,
    prediction: np.ndarray,
    target_count: int,
    prediction_count: int,
) -> np.ndarray:
    if target_count == 0 or prediction_count == 0:
        return np.zeros((target_count, prediction_count), dtype=np.float64)
    stride = prediction_count + 1
    pair_index = target.astype(np.int64) * stride + prediction.astype(np.int64)
    intersections = np.bincount(
        pair_index.reshape(-1), minlength=(target_count + 1) * stride
    ).reshape(target_count + 1, stride)[1:, 1:]
    target_area = np.bincount(target.reshape(-1), minlength=target_count + 1)[1:]
    prediction_area = np.bincount(
        prediction.reshape(-1), minlength=prediction_count + 1
    )[1:]
    unions = target_area[:, None] + prediction_area[None, :] - intersections
    return np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )


def instance_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    iou_threshold: float = 0.5,
) -> InstanceMetrics:
    """Compute one-to-one PQ at the conventional strict IoU 0.5 threshold."""
    if target.shape != prediction.shape:
        raise ValueError(f"Label shapes differ: {target.shape} and {prediction.shape}")
    if not 0 < iou_threshold < 1:
        raise ValueError("IoU threshold must lie strictly between zero and one")
    target_labels, target_count = _canonical_positive_labels(target)
    prediction_labels, prediction_count = _canonical_positive_labels(prediction)
    iou = _intersection_over_union(
        target_labels, prediction_labels, target_count, prediction_count
    )
    if iou.size:
        target_indices, prediction_indices = linear_sum_assignment(-iou)
        matched_iou = iou[target_indices, prediction_indices]
        accepted = matched_iou > iou_threshold
        matched_sum = float(matched_iou[accepted].sum())
        true_positives = int(accepted.sum())
    else:
        matched_sum = 0.0
        true_positives = 0
    false_positives = prediction_count - true_positives
    false_negatives = target_count - true_positives
    recognition_denominator = (
        true_positives + 0.5 * false_positives + 0.5 * false_negatives
    )
    recognition_quality = (
        true_positives / recognition_denominator
        if recognition_denominator > 0
        else 1.0
    )
    segmentation_quality = (
        matched_sum / true_positives
        if true_positives > 0
        else (1.0 if target_count == prediction_count == 0 else 0.0)
    )
    panoptic_quality = (
        matched_sum / recognition_denominator
        if recognition_denominator > 0
        else 1.0
    )
    absolute_count_error = abs(prediction_count - target_count)
    target_foreground = target_labels > 0
    prediction_foreground = prediction_labels > 0
    intersection = np.logical_and(target_foreground, prediction_foreground).sum()
    foreground_denominator = target_foreground.sum() + prediction_foreground.sum()
    foreground_dice = (
        float(2 * intersection / foreground_denominator)
        if foreground_denominator > 0
        else 1.0
    )
    return InstanceMetrics(
        panoptic_quality=panoptic_quality,
        segmentation_quality=segmentation_quality,
        recognition_quality=recognition_quality,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        predicted_count=prediction_count,
        target_count=target_count,
        absolute_count_error=absolute_count_error,
        percentage_count_error=absolute_count_error / max(target_count, 1),
        foreground_dice=foreground_dice,
    )


def resample_labels(labels: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize hard labels with nearest-neighbor interpolation only."""
    value = np.asarray(labels)
    if value.ndim < 2:
        raise ValueError("Labels must have at least two spatial dimensions")
    leading_shape = value.shape[:-2]
    tensor = torch.from_numpy(value.astype(np.float32, copy=False)).reshape(
        -1, 1, *value.shape[-2:]
    )
    resized = functional.interpolate(tensor, size=output_shape, mode="nearest")
    return resized[:, 0].to(torch.int32).numpy().reshape(*leading_shape, *output_shape)


def valid_region_labels(
    target: np.ndarray,
    prediction: np.ndarray,
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen border rule, then return the central valid region.

    Target instances must be completely contained. Predicted instances count
    when their full-image centroid lies inside the valid rectangle.
    """
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Target and prediction must be same-shape 2D label images")
    height, width = target.shape
    if margin < 0 or 2 * margin >= min(height, width):
        raise ValueError("Valid-region margin is outside the label image")

    keep_target = np.zeros(target.shape, dtype=np.int32)
    for object_id in np.unique(target[target > 0]):
        yy, xx = np.nonzero(target == object_id)
        contained = (
            yy.min() >= margin
            and yy.max() < height - margin
            and xx.min() >= margin
            and xx.max() < width - margin
        )
        if contained:
            keep_target[target == object_id] = int(object_id)

    keep_prediction = np.zeros(prediction.shape, dtype=np.int32)
    for object_id in np.unique(prediction[prediction > 0]):
        yy, xx = np.nonzero(prediction == object_id)
        center_y = float(yy.mean())
        center_x = float(xx.mean())
        if (
            margin <= center_y < height - margin
            and margin <= center_x < width - margin
        ):
            keep_prediction[prediction == object_id] = int(object_id)

    region = np.s_[margin : height - margin, margin : width - margin]
    return keep_target[region], keep_prediction[region]


def grouped_bootstrap_difference(
    source_ids: np.ndarray,
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    replicates: int = 1000,
    seed: int = 20260901,
) -> dict[str, float | int]:
    """Bootstrap a paired mean difference by resampling whole source images."""
    source_ids = np.asarray(source_ids)
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    if not (
        source_ids.shape == candidate_values.shape == reference_values.shape
    ):
        raise ValueError("Grouped bootstrap inputs must have identical shapes")
    if source_ids.ndim != 1 or source_ids.size == 0:
        raise ValueError("Grouped bootstrap inputs must be non-empty vectors")
    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    unique_sources = np.unique(source_ids)
    source_differences = np.asarray(
        [
            (candidate_values[source_ids == source] - reference_values[source_ids == source]).mean()
            for source in unique_sources
        ]
    )
    generator = np.random.default_rng(seed)
    sampled_indices = generator.integers(
        0, len(unique_sources), size=(replicates, len(unique_sources))
    )
    bootstrap_means = source_differences[sampled_indices].mean(axis=1)
    return {
        "source_count": int(len(unique_sources)),
        "replicates": replicates,
        "mean_difference": float(source_differences.mean()),
        "ci_lower_95": float(np.quantile(bootstrap_means, 0.025)),
        "ci_upper_95": float(np.quantile(bootstrap_means, 0.975)),
    }
