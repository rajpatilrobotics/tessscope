"""Served two-Tesseract training utilities for matched design runs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from tessscope.data.bbbc039 import (
    ObjectNormalization,
    PreparedPatch,
    prepare_patch,
    records_for_split,
)
from tessscope.observer.calibration import load_frozen_transform
from tessscope.optimization.objectives import image_fidelity_loss
from tessscope.optimization.schedule import BatchSpec


@dataclass(frozen=True)
class TrainingBatch:
    object_batch: np.ndarray
    instance_labels: np.ndarray
    centers_yx: np.ndarray
    valid_objects: np.ndarray
    depths_um: np.ndarray
    patch_ids: tuple[str, ...]


def _record_lookup():
    return {record.image_id: record for record in records_for_split("training")}


def materialize_batch(spec: BatchSpec) -> TrainingBatch:
    """Load one frozen schedule item with all static observer supervision."""
    records = _record_lookup()
    normalization = ObjectNormalization(125.0, 1642.0)
    patches: list[PreparedPatch] = [
        prepare_patch(records[key.image_id], key.origin_yx, normalization)
        for key in spec.patches
    ]
    return TrainingBatch(
        object_batch=np.stack([patch.object_image for patch in patches]).astype(np.float32),
        instance_labels=np.stack([patch.instance_labels for patch in patches]).astype(
            np.int32
        ),
        centers_yx=np.stack([patch.centers_yx for patch in patches]).astype(np.float32),
        valid_objects=np.stack([patch.valid_objects for patch in patches]).astype(np.uint8),
        depths_um=np.asarray(spec.depths_um, dtype=np.float32),
        patch_ids=tuple(
            f"{patch.image_id}:{patch.origin_yx[0]}:{patch.origin_yx[1]}"
            for patch in patches
        ),
    )


def observer_static_payload(
    batch: TrainingBatch,
    *,
    backward_mode: str = "exact",
    surrogate_scale: float = 1.0,
) -> dict[str, object]:
    transform = load_frozen_transform()
    return {
        "instance_labels": batch.instance_labels,
        "centers_yx": batch.centers_yx,
        "valid_objects": batch.valid_objects,
        "transform_mode": transform.mode,
        "transform_offset": transform.offset,
        "transform_scale": transform.scale,
        "backward_mode": backward_mode,
        "surrogate_scale": surrogate_scale,
    }


def task_value_and_gradient(
    optics: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batch: TrainingBatch,
    *,
    backward_mode: str = "exact",
    surrogate_scale: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Return task value and served phase gradient for one matched batch."""
    optics_static = {
        "object_batch": batch.object_batch,
        "depths_um": batch.depths_um,
        "expected_photons": 100.0,
    }
    observer_static = observer_static_payload(
        batch,
        backward_mode=backward_mode,
        surrogate_scale=surrogate_scale,
    )

    def objective(value: jax.Array) -> jax.Array:
        formed = apply_tesseract(
            optics, {"phase_parameters": value, **optics_static}
        )
        return apply_tesseract(
            observer,
            {
                "sensor": formed["sensor"],
                "phase_coefficients": formed["phase_coefficients"],
                **observer_static,
            },
        )["task_loss"]

    value, gradient = jax.value_and_grad(objective)(
        jnp.asarray(parameters, dtype=jnp.float32)
    )
    return float(value), np.asarray(gradient, dtype=np.float32)


def image_value_and_gradient(
    optics: Tesseract,
    parameters: np.ndarray,
    batch: TrainingBatch,
    clear_focus_reference: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return matched image-fidelity value and served-optics phase gradient."""
    optics_static = {
        "object_batch": batch.object_batch,
        "depths_um": batch.depths_um,
        "expected_photons": 100.0,
    }

    def objective(value: jax.Array) -> jax.Array:
        formed = apply_tesseract(
            optics, {"phase_parameters": value, **optics_static}
        )
        return image_fidelity_loss(
            formed["sensor"],
            jnp.asarray(clear_focus_reference, dtype=jnp.float32),
            formed["phase_coefficients"],
        )

    value, gradient = jax.value_and_grad(objective)(
        jnp.asarray(parameters, dtype=jnp.float32)
    )
    return float(value), np.asarray(gradient, dtype=np.float32)


def observer_sensor_gradient_norms(
    observer: Tesseract,
    sensor: np.ndarray,
    phase_coefficients: np.ndarray,
    batch: TrainingBatch,
) -> tuple[float, float]:
    """Return exact and unit-scale proxy sensor-gradient norms for calibration."""
    sensor_value = jnp.asarray(sensor, dtype=jnp.float32)
    coefficient_value = jnp.asarray(phase_coefficients, dtype=jnp.float32)

    def loss(value: jax.Array, mode: str) -> jax.Array:
        return apply_tesseract(
            observer,
            {
                "sensor": value,
                "phase_coefficients": coefficient_value,
                **observer_static_payload(batch, backward_mode=mode),
            },
        )["task_loss"]

    exact = jax.grad(lambda value: loss(value, "exact"))(sensor_value)
    proxy = jax.grad(lambda value: loss(value, "surrogate"))(sensor_value)
    return float(jnp.linalg.norm(exact)), float(jnp.linalg.norm(proxy))
