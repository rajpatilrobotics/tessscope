"""Served first-exposure → stage-action → corrected-exposure objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from tessscope.v2.data.patches import PreparedV2Patch
from tessscope.v2.optimization.served import DEPTHS, V2SystemCalibration


@dataclass(frozen=True)
class ClosedLoopWeights:
    """Transparent weights for one pre-registered v2.3 objective profile."""

    first_segmentation: float
    final_segmentation: float = 1.0
    normalized_residual_depth_squared: float = 0.10
    normalized_stage_action_squared: float = 0.01
    fixed_second_exposure_cost: float = 0.01

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("Closed-loop weights must be nonnegative")


OBJECTIVE_PROFILES = {
    "balanced": ClosedLoopWeights(first_segmentation=0.35),
    "first_heavy": ClosedLoopWeights(first_segmentation=0.75),
    "action_heavy": ClosedLoopWeights(
        first_segmentation=0.35,
        normalized_residual_depth_squared=0.25,
    ),
}


@dataclass(frozen=True)
class ClosedLoopBatch:
    """Two autofocus supports and one supervised query for two exposures."""

    objects: np.ndarray
    instance_labels: np.ndarray
    centers_yx: np.ndarray
    valid_objects: np.ndarray
    patch_ids: tuple[str, str, str]
    first_noise_standard_normal: np.ndarray
    second_noise_standard_normal: np.ndarray


def materialize_closed_loop_batch(
    patches: list[PreparedV2Patch] | tuple[PreparedV2Patch, ...],
    *,
    noise_seed: int | None,
) -> ClosedLoopBatch:
    """Create independent first/second training noise without touching test data."""
    if len(patches) != 3:
        raise ValueError("A closed-loop batch requires two supports and one query")
    if noise_seed is None:
        first_noise = np.zeros((3, 7, 256, 256), dtype=np.float32)
        second_noise = np.zeros((1, 7, 256, 256), dtype=np.float32)
    else:
        generator = np.random.default_rng(noise_seed)
        first_noise = generator.normal(size=(3, 7, 256, 256)).astype(np.float32)
        second_noise = generator.normal(size=(1, 7, 256, 256)).astype(np.float32)
    query = patches[2]
    return ClosedLoopBatch(
        objects=np.stack([patch.object_image for patch in patches]).astype(np.float32),
        instance_labels=query.instance_labels[None].astype(np.int32),
        centers_yx=query.centers_yx[None].astype(np.float32),
        valid_objects=query.valid_objects[None].astype(np.uint8),
        patch_ids=tuple(
            f"{patch.field_id}:{patch.origin_yx[0]}:{patch.origin_yx[1]}"
            for patch in patches
        ),
        first_noise_standard_normal=first_noise,
        second_noise_standard_normal=second_noise,
    )


def nondominated_closed_loop_names(rows: list[dict]) -> set[str]:
    """Return endpoints not dominated on first loss, final loss, and residual MAE."""
    names = set()
    metrics = (
        "first_segmentation_loss",
        "final_segmentation_loss",
        "residual_mae_um",
    )
    for candidate in rows:
        dominated = any(
            other["name"] != candidate["name"]
            and all(other[metric] <= candidate[metric] for metric in metrics)
            and any(other[metric] < candidate[metric] for metric in metrics)
            for other in rows
        )
        if not dominated:
            names.add(candidate["name"])
    return names


def loss_from_terms(
    first_segmentation_loss: jax.Array,
    final_segmentation_loss: jax.Array,
    stage_action_um: jax.Array,
    true_depth_um: jax.Array,
    weights: ClosedLoopWeights,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Combine the pre-registered objective and return transparent diagnostics."""
    residual_depth = true_depth_um + stage_action_um
    residual_squared = jnp.mean(jnp.square(residual_depth / 6.0))
    action_squared = jnp.mean(jnp.square(stage_action_um / 6.0))
    value = (
        weights.first_segmentation * first_segmentation_loss
        + weights.final_segmentation * final_segmentation_loss
        + weights.normalized_residual_depth_squared * residual_squared
        + weights.normalized_stage_action_squared * action_squared
        + weights.fixed_second_exposure_cost
    )
    return value, (residual_depth, residual_squared, action_squared)


def _graph(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: jax.Array,
    batch: ClosedLoopBatch,
    calibration: V2SystemCalibration,
    weights: ClosedLoopWeights,
    gradient_mode: Literal["exact", "stop_stage"],
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    optics_common = {
        "expected_photons": calibration.expected_photons,
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
    first = apply_tesseract(
        optics,
        {
            "phase_parameters": parameters,
            "object_batch": batch.objects,
            "depths_um": DEPTHS,
            "noise_standard_normal": batch.first_noise_standard_normal,
            **optics_common,
        },
    )
    focused = apply_tesseract(
        autofocus,
        {
            "support_sensor": first["sensor"][:2].reshape(-1, 256, 256),
            "support_depth_um": np.tile(DEPTHS, 2),
            "query_sensor": first["sensor"][2],
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
            "sensor": first["sensor"][2:3],
            "phase_coefficients": first["phase_coefficients"],
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
    second = apply_tesseract(
        optics,
        {
            "phase_parameters": parameters,
            "object_batch": batch.objects[2:3],
            "depths_um": residual_depth,
            "noise_standard_normal": batch.second_noise_standard_normal,
            **optics_common,
        },
    )
    final_observation = apply_tesseract(
        observer,
        {
            "sensor": second["sensor"],
            "phase_coefficients": second["phase_coefficients"],
            **observer_common,
        },
    )
    value, loss_terms = loss_from_terms(
        first_observation["task_loss"],
        final_observation["task_loss"],
        residual_action,
        jnp.asarray(DEPTHS),
        weights,
    )
    actual_residual, residual_squared, action_squared = loss_terms
    return value, (
        first_observation["task_loss"],
        final_observation["task_loss"],
        focused["focus_mse"],
        jnp.mean(jnp.abs(actual_residual)),
        residual_squared,
        action_squared,
        first["photon_mean"],
        second["photon_mean"],
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


def closed_loop_value_and_gradient(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batch: ClosedLoopBatch,
    calibration: V2SystemCalibration,
    weights: ClosedLoopWeights,
    *,
    gradient_mode: Literal["exact", "stop_stage"] = "exact",
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return the served feedback-loop value, phase gradient, and branch values."""
    if gradient_mode not in {"exact", "stop_stage"}:
        raise ValueError(f"Unknown closed-loop gradient mode: {gradient_mode}")

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
    return float(value), np.asarray(gradient), _diagnostics(auxiliary)


def closed_loop_value(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batch: ClosedLoopBatch,
    calibration: V2SystemCalibration,
    weights: ClosedLoopWeights,
    *,
    gradient_mode: Literal["exact", "stop_stage"] = "exact",
) -> tuple[float, dict[str, float]]:
    """Return the forward-identical objective without requesting a VJP."""
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
