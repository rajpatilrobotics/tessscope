"""Shared three-Tesseract joint batches, calibration, and exact served gradients."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from tessscope.v2.data.bbbc006 import PROJECT_ROOT, records_for_split
from tessscope.v2.data.patches import PreparedV2Patch, patch_origins, prepare_patch
from tessscope.v2.data.preprocess import (
    accepted_registration_fields,
    load_global_normalization,
)
from tessscope.v2.optimization.objective import JointObjectiveWeights

DEPTHS = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
CALIBRATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "gate3"
    / "clear-calibration-offset.json"
)
TRAINING_CALIBRATION_PATH = (
    PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-training-calibration.json"
)


@dataclass(frozen=True)
class V2SystemCalibration:
    """Frozen training/validation optical constants used by every matched design."""

    exposure_gain: float
    depth_scale: float
    axial_offset_um: float
    ridge_lambda: float = 100.0
    expected_photons: float = 200.0

    @classmethod
    def load(cls) -> V2SystemCalibration:
        optical = json.loads(CALIBRATION_PATH.read_text())
        intensity = json.loads(TRAINING_CALIBRATION_PATH.read_text())
        if not optical["passed"]:
            raise ValueError("Cannot optimize before the clear calibration gate passes")
        return cls(
            exposure_gain=float(intensity["exposure"]["gain"]),
            depth_scale=float(optical["model"]["fitted_depth_scale"]),
            axial_offset_um=float(optical["model"]["fitted_axial_offset_um"]),
        )


@dataclass(frozen=True)
class ServedJointBatch:
    """Two autofocus supports plus one supervised query/segmentation patch."""

    objects: np.ndarray
    instance_labels: np.ndarray
    centers_yx: np.ndarray
    valid_objects: np.ndarray
    patch_ids: tuple[str, str, str]
    noise_standard_normal: np.ndarray


def collect_patches(split: str, count: int, *, seed: int) -> list[PreparedV2Patch]:
    """Collect a deterministic well-split patch set with at least one valid instance."""
    if split not in {"training", "validation"}:
        raise ValueError("Optimization patch collection excludes the locked test split")
    normalization = load_global_normalization()
    accepted = accepted_registration_fields()
    candidates = [
        (record, origin)
        for record in records_for_split(split)
        if record.field_id in accepted
        for origin in patch_origins(split)
    ]
    ranked = sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"v2-patches:{seed}:{item[0].field_id}:{item[1]}".encode()
        ).digest(),
    )
    selected = []
    selected_wells: set[str] = set()
    for record, origin in ranked:
        if record.well in selected_wells:
            continue
        patch = prepare_patch(record, origin, normalization)
        if np.count_nonzero(patch.valid_objects) == 0:
            continue
        selected.append(patch)
        selected_wells.add(record.well)
        if len(selected) == count:
            return selected
    raise ValueError(f"Could not collect {count} valid {split} patches")


def materialize_joint_batch(
    patches: list[PreparedV2Patch] | tuple[PreparedV2Patch, ...],
    *,
    noise_seed: int | None,
) -> ServedJointBatch:
    """Materialize exactly two support patches and one query patch."""
    if len(patches) != 3:
        raise ValueError("A served joint batch requires exactly three patches")
    query = patches[2]
    noise = (
        np.zeros((3, 7, 256, 256), dtype=np.float32)
        if noise_seed is None
        else np.random.default_rng(noise_seed)
        .normal(size=(3, 7, 256, 256))
        .astype(np.float32)
    )
    return ServedJointBatch(
        objects=np.stack([patch.object_image for patch in patches]).astype(np.float32),
        instance_labels=query.instance_labels[None].astype(np.int32),
        centers_yx=query.centers_yx[None].astype(np.float32),
        valid_objects=query.valid_objects[None].astype(np.uint8),
        patch_ids=tuple(
            f"{patch.field_id}:{patch.origin_yx[0]}:{patch.origin_yx[1]}"
            for patch in patches
        ),
        noise_standard_normal=noise,
    )


def joint_value_and_gradient(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batch: ServedJointBatch,
    calibration: V2SystemCalibration,
    weights: JointObjectiveWeights,
    *,
    gradient_mode: Literal["exact", "stop_focus"] = "exact",
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return one served joint value/gradient and transparent branch diagnostics."""
    if gradient_mode not in {"exact", "stop_focus"}:
        raise ValueError(f"Unknown joint gradient mode: {gradient_mode}")
    optics_static = {
        "object_batch": batch.objects,
        "depths_um": DEPTHS,
        "noise_standard_normal": batch.noise_standard_normal,
        "expected_photons": calibration.expected_photons,
        "exposure_gain": calibration.exposure_gain,
        "depth_scale": calibration.depth_scale,
        "axial_offset_um": calibration.axial_offset_um,
    }
    autofocus_static = {
        "support_depth_um": np.tile(DEPTHS, 2),
        "query_depth_um": DEPTHS,
        "ridge_lambda": calibration.ridge_lambda,
        "radial_bins": 10,
        "angular_bins": 12,
    }
    observer_static = {
        "instance_labels": batch.instance_labels,
        "centers_yx": batch.centers_yx,
        "valid_objects": batch.valid_objects,
        "transform_mode": "affine",
        "transform_offset": 0.0,
        "transform_scale": calibration.exposure_gain,
        "backward_mode": "exact",
        "surrogate_scale": 1.0,
    }

    def objective(value: jax.Array):
        formed = apply_tesseract(
            optics, {"phase_parameters": value, **optics_static}
        )
        focused = apply_tesseract(
            autofocus,
            {
                "support_sensor": formed["sensor"][:2].reshape(-1, 256, 256),
                "query_sensor": formed["sensor"][2],
                **autofocus_static,
            },
        )
        segmented = apply_tesseract(
            observer,
            {
                "sensor": formed["sensor"][2:3],
                "phase_coefficients": formed["phase_coefficients"],
                **observer_static,
            },
        )
        focus_mse = focused["focus_mse"]
        if gradient_mode == "stop_focus":
            focus_mse = jax.lax.stop_gradient(focus_mse)
        joint = weights.combine(segmented["task_loss"], focus_mse)
        return joint, (
            segmented["task_loss"],
            focused["focus_mse"],
            formed["photon_mean"],
        )

    (value, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(
        jnp.asarray(parameters, dtype=jnp.float32)
    )
    return (
        float(value),
        np.asarray(gradient, dtype=np.float32),
        {
            "segmentation_loss": float(auxiliary[0]),
            "focus_mse": float(auxiliary[1]),
            "photon_mean": float(auxiliary[2]),
        },
    )


def joint_value(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batch: ServedJointBatch,
    calibration: V2SystemCalibration,
    weights: JointObjectiveWeights,
) -> tuple[float, dict[str, float]]:
    """Return the same served forward objective without requesting any VJP."""
    formed = apply_tesseract(
        optics,
        {
            "phase_parameters": jnp.asarray(parameters, dtype=jnp.float32),
            "object_batch": batch.objects,
            "depths_um": DEPTHS,
            "noise_standard_normal": batch.noise_standard_normal,
            "expected_photons": calibration.expected_photons,
            "exposure_gain": calibration.exposure_gain,
            "depth_scale": calibration.depth_scale,
            "axial_offset_um": calibration.axial_offset_um,
        },
    )
    focused = apply_tesseract(
        autofocus,
        {
            "support_sensor": formed["sensor"][:2].reshape(-1, 256, 256),
            "support_depth_um": np.tile(DEPTHS, 2),
            "query_sensor": formed["sensor"][2],
            "query_depth_um": DEPTHS,
            "ridge_lambda": calibration.ridge_lambda,
            "radial_bins": 10,
            "angular_bins": 12,
        },
    )
    segmented = apply_tesseract(
        observer,
        {
            "sensor": formed["sensor"][2:3],
            "phase_coefficients": formed["phase_coefficients"],
            "instance_labels": batch.instance_labels,
            "centers_yx": batch.centers_yx,
            "valid_objects": batch.valid_objects,
            "transform_mode": "affine",
            "transform_offset": 0.0,
            "transform_scale": calibration.exposure_gain,
            "backward_mode": "exact",
            "surrogate_scale": 1.0,
        },
    )
    value = weights.combine(segmented["task_loss"], focused["focus_mse"])
    return float(value), {
        "segmentation_loss": float(segmented["task_loss"]),
        "focus_mse": float(focused["focus_mse"]),
        "photon_mean": float(formed["photon_mean"]),
    }
