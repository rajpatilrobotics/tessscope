"""Differentiable sequential sensing-mask → action → capture-mask graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from tessscope.v2.optimization.served import DEPTHS, V2SystemCalibration
from tessscope.v2_3.closed_loop import ClosedLoopBatch

GradientMode = Literal["exact", "stop_stage"]


@dataclass(frozen=True)
class TwoMaskWeights:
    """Transparent linear objective weights for two-mask optimization."""

    first_segmentation: float = 0.0
    final_segmentation: float = 1.0
    normalized_residual_depth_squared: float = 0.0

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("Two-mask weights must be nonnegative")


FINAL_WEIGHTS = TwoMaskWeights()
FIRST_WEIGHTS = TwoMaskWeights(first_segmentation=1.0, final_segmentation=0.0)
RESIDUAL_WEIGHTS = TwoMaskWeights(
    final_segmentation=0.0,
    normalized_residual_depth_squared=1.0,
)


def join_two_mask_parameters(
    sensing_parameters: np.ndarray,
    capture_parameters: np.ndarray,
) -> np.ndarray:
    """Join two B7 unconstrained vectors in the frozen sensing/capture order."""
    sensing = np.asarray(sensing_parameters, dtype=np.float32)
    capture = np.asarray(capture_parameters, dtype=np.float32)
    if sensing.shape != (7,) or capture.shape != (7,):
        raise ValueError("Two-mask parameters require two seven-value B7 vectors")
    return np.concatenate([sensing, capture]).astype(np.float32)


def split_two_mask_parameters(parameters: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a 14-value vector into its sensing and capture B7 masks."""
    value = jnp.asarray(parameters, dtype=jnp.float32)
    if value.shape != (14,):
        raise ValueError(f"Expected 14 two-mask parameters, got {value.shape}")
    return value[:7], value[7:]


def _graph(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: jax.Array,
    batch: ClosedLoopBatch,
    calibration: V2SystemCalibration,
    weights: TwoMaskWeights,
    gradient_mode: GradientMode,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    sensing_parameters, capture_parameters = split_two_mask_parameters(parameters)
    optics_common = {
        "expected_photons": 200.0,
        "exposure_gain": calibration.exposure_gain,
        "depth_scale": calibration.depth_scale,
        "axial_offset_um": calibration.axial_offset_um,
    }
    observer_common = {
        "instance_labels": batch.instance_labels,
        "centers_yx": batch.centers_yx,
        "valid_objects": batch.valid_objects,
        "transform_mode": "affine",
        "transform_offset": 0.0,
        "transform_scale": calibration.exposure_gain,
        "backward_mode": "exact",
        "surrogate_scale": 1.0,
    }
    sensing = apply_tesseract(
        optics,
        {
            "phase_parameters": sensing_parameters,
            "object_batch": batch.objects,
            "depths_um": DEPTHS,
            "noise_standard_normal": batch.first_noise_standard_normal,
            **optics_common,
        },
    )
    focused = apply_tesseract(
        autofocus,
        {
            "support_sensor": sensing["sensor"][:2].reshape(-1, 256, 256),
            "support_depth_um": np.tile(DEPTHS, 2),
            "query_sensor": sensing["sensor"][2],
            "query_depth_um": DEPTHS,
            "ridge_lambda": calibration.ridge_lambda,
            "radial_bins": 10,
            "angular_bins": 12,
            "stage_bound_um": 6.0,
        },
    )
    first_observation = apply_tesseract(
        observer,
        {
            "sensor": sensing["sensor"][2:3],
            "phase_coefficients": sensing["phase_coefficients"],
            **observer_common,
        },
    )
    stage_action = focused["stage_action_um"]
    residual_action = (
        stage_action
        if gradient_mode == "exact"
        else jax.lax.stop_gradient(stage_action)
    )
    residual_depth = jnp.asarray(DEPTHS) + residual_action
    capture = apply_tesseract(
        optics,
        {
            "phase_parameters": capture_parameters,
            "object_batch": batch.objects[2:3],
            "depths_um": residual_depth,
            "noise_standard_normal": batch.second_noise_standard_normal,
            **optics_common,
        },
    )
    final_observation = apply_tesseract(
        observer,
        {
            "sensor": capture["sensor"],
            "phase_coefficients": capture["phase_coefficients"],
            **observer_common,
        },
    )
    actual_residual = jnp.asarray(DEPTHS) + stage_action
    residual_squared = jnp.mean(jnp.square(actual_residual / 6.0))
    value = (
        weights.first_segmentation * first_observation["task_loss"]
        + weights.final_segmentation * final_observation["task_loss"]
        + weights.normalized_residual_depth_squared * residual_squared
    )
    return value, (
        first_observation["task_loss"],
        final_observation["task_loss"],
        focused["focus_mse"],
        jnp.mean(jnp.abs(actual_residual)),
        residual_squared,
        jnp.mean(jnp.square(stage_action / 6.0)),
        sensing["photon_mean"],
        capture["photon_mean"],
    )


def _diagnostics(auxiliary: tuple[jax.Array, ...]) -> dict[str, float]:
    names = (
        "first_segmentation_loss",
        "final_segmentation_loss",
        "focus_mse",
        "residual_mae_um",
        "normalized_residual_depth_squared",
        "normalized_stage_action_squared",
        "first_photon_mean",
        "second_photon_mean",
    )
    return {name: float(value) for name, value in zip(names, auxiliary, strict=True)}


def two_mask_value_and_gradient(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batch: ClosedLoopBatch,
    calibration: V2SystemCalibration,
    weights: TwoMaskWeights = FINAL_WEIGHTS,
    *,
    gradient_mode: GradientMode = "exact",
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return one two-mask objective, exact 14-vector gradient, and diagnostics."""
    if gradient_mode not in {"exact", "stop_stage"}:
        raise ValueError(f"Unknown two-mask gradient mode: {gradient_mode}")

    def objective(value: jax.Array):
        return _graph(
            optics,
            autofocus,
            observer,
            value,
            batch,
            calibration,
            weights,
            gradient_mode,
        )

    (value, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(
        jnp.asarray(parameters, dtype=jnp.float32)
    )
    return float(value), np.asarray(gradient, dtype=np.float32), _diagnostics(auxiliary)


def two_mask_value(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batch: ClosedLoopBatch,
    calibration: V2SystemCalibration,
    weights: TwoMaskWeights = FINAL_WEIGHTS,
    *,
    gradient_mode: GradientMode = "exact",
) -> tuple[float, dict[str, float]]:
    """Return the forward two-mask objective without requesting a VJP."""
    if gradient_mode not in {"exact", "stop_stage"}:
        raise ValueError(f"Unknown two-mask gradient mode: {gradient_mode}")
    value, auxiliary = _graph(
        optics,
        autofocus,
        observer,
        jnp.asarray(parameters, dtype=jnp.float32),
        batch,
        calibration,
        weights,
        gradient_mode,
    )
    return float(value), _diagnostics(auxiliary)


def average_two_mask_forward(
    services: tuple[Tesseract, Tesseract, Tesseract],
    parameters: np.ndarray,
    batches: tuple[ClosedLoopBatch, ...] | list[ClosedLoopBatch],
    calibration: V2SystemCalibration,
    *,
    gradient_mode: GradientMode = "exact",
) -> dict[str, float]:
    """Average all transparent terms over a fixed partition's four batches."""
    rows = [
        two_mask_value(
            *services,
            parameters,
            batch,
            calibration,
            FINAL_WEIGHTS,
            gradient_mode=gradient_mode,
        )[1]
        for batch in batches
    ]
    if not rows:
        raise ValueError("At least one two-mask batch is required")
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
