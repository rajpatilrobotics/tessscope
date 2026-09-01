"""Deterministic BBBC006 object/label patches for v2 optical co-design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from tessscope.data.bbbc039 import (
    padded_centers,
    resample_labels_for_observer,
    select_patch_instances,
    split_patch,
)
from tessscope.v2.data.bbbc006 import BBBC006StackRecord, load_reference_labels
from tessscope.v2.data.preprocess import GlobalNormalization

TRAINING_PATCH_ORIGINS = ((0, 0), (0, 440), (264, 0), (264, 440))
VALIDATION_PATCH_ORIGINS = ((0, 220), (264, 220))


@dataclass(frozen=True)
class PreparedV2Patch:
    """One globally normalized optical object plus fixed observer supervision."""

    field_id: str
    well: str
    site: int
    split: str
    origin_yx: tuple[int, int]
    object_image: np.ndarray
    instance_labels: np.ndarray
    centers_yx: np.ndarray
    valid_objects: np.ndarray


def patch_origins(split: str) -> tuple[tuple[int, int], ...]:
    """Return fixed non-test crop origins for optimization/validation."""
    if split == "training":
        return TRAINING_PATCH_ORIGINS
    if split == "validation":
        return VALIDATION_PATCH_ORIGINS
    raise ValueError("V2 optimization patch origins exclude the locked test split")


def prepare_patch(
    record: BBBC006StackRecord,
    origin_yx: tuple[int, int],
    normalization: GlobalNormalization,
    *,
    valid_margin: int = 48,
    maximum_instances: int = 32,
) -> PreparedV2Patch:
    """Prepare the z16 object proxy and automated reference instances."""
    with Image.open(record.image_paths[3]) as image:
        raw_focus = np.asarray(image)
    raw_patch = split_patch(raw_focus, origin_yx)
    label_patch = split_patch(load_reference_labels(record), origin_yx)
    key = f"v2:{record.field_id}:{origin_yx[0]}:{origin_yx[1]}"
    selected = select_patch_instances(
        label_patch,
        key=key,
        valid_margin=valid_margin,
        maximum_instances=maximum_instances,
    )
    observer_labels = resample_labels_for_observer(selected)
    centers, valid = padded_centers(
        observer_labels, maximum_instances=maximum_instances
    )
    return PreparedV2Patch(
        field_id=record.field_id,
        well=record.well,
        site=record.site,
        split=record.split,
        origin_yx=origin_yx,
        object_image=normalization.apply(raw_patch).astype(np.float32),
        instance_labels=observer_labels,
        centers_yx=centers,
        valid_objects=valid,
    )
