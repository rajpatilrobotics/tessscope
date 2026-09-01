"""Frozen global normalization and registered-crop helpers for BBBC006 v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tessscope.v2.data.bbbc006 import PROJECT_ROOT, BBBC006StackRecord, load_stack
from tessscope.v2.data.registration import apply_translation

REGISTRATION_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-registrations.json"
)
CALIBRATION_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-training-calibration.json"
)


@dataclass(frozen=True)
class GlobalNormalization:
    """Training-only affine transform shared by every image and split."""

    offset: float
    scale: float

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.scale <= 0:
            raise ValueError("Global normalization scale must be positive")
        return (np.asarray(image, dtype=np.float64) - self.offset) / self.scale


def center_crop(image: np.ndarray, size: int = 256) -> np.ndarray:
    """Take the same centered square field from every image."""
    value = np.asarray(image)
    if value.ndim < 2 or size <= 0 or size > min(value.shape[-2:]):
        raise ValueError(f"Cannot take a {size}px center crop from {value.shape}")
    y_start = (value.shape[-2] - size) // 2
    x_start = (value.shape[-1] - size) // 2
    return value[..., y_start : y_start + size, x_start : x_start + size]


def load_global_normalization(
    path: Path = CALIBRATION_MANIFEST,
) -> GlobalNormalization:
    """Load frozen train-only normalization constants."""
    payload = json.loads(path.read_text())
    if payload["status"] != "complete_training_only":
        raise ValueError("BBBC006 training calibration is not complete")
    normalization = payload["normalization"]
    return GlobalNormalization(
        offset=float(normalization["offset"]), scale=float(normalization["scale"])
    )


def load_registration_table(
    path: Path = REGISTRATION_MANIFEST,
) -> dict[tuple[str, int, int], tuple[float, float]]:
    """Load train/validation translations indexed by field and z-plane."""
    payload = json.loads(path.read_text())
    if payload["status"] != "complete_train_validation":
        raise ValueError("BBBC006 registration manifest is not complete")
    return {
        (row["well"], int(row["site"]), int(row["z_plane"])): tuple(
            float(value) for value in row["shift_yx_px"]
        )
        for row in payload["registrations"]
        if row["status"] == "accepted"
    }


def accepted_registration_fields(
    path: Path = REGISTRATION_MANIFEST,
) -> set[str]:
    """Return fields that passed registration at every selected z-plane."""
    payload = json.loads(path.read_text())
    if payload["status"] != "complete_train_validation":
        raise ValueError("BBBC006 registration manifest is not complete")
    return {
        row["field_id"]
        for row in payload["registrations"]
        if row["field_accepted"]
    }


def registered_normalized_crop(
    record: BBBC006StackRecord,
    normalization: GlobalNormalization,
    registrations: dict[tuple[str, int, int], tuple[float, float]],
    *,
    size: int = 256,
) -> np.ndarray:
    """Load one stack, apply frozen translations, normalize globally, and crop."""
    stack = load_stack(record)
    registered = []
    for plane, image in zip(range(13, 20), stack, strict=True):
        key = (record.well, record.site, plane)
        if key not in registrations:
            raise KeyError(f"Missing frozen registration for {key}")
        registered.append(apply_translation(image, registrations[key]))
    return center_crop(normalization.apply(np.stack(registered)), size=size)
