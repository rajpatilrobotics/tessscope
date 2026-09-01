"""Frozen BBBC039 records, mask decoding, and optimization patches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from skimage.measure import label as connected_components

from tessscope.data.decontam import BBBC039_ROOT, MANIFEST_ROOT, discover_bbbc039

TRAIN_PATCH_ORIGINS = ((0, 0), (0, 440), (264, 0), (264, 440))
VALIDATION_PATCH_ORIGINS = ((0, 220), (264, 220))
PATCH_SIZE = 256
BIOLOGICAL_SAMPLING_UM = 0.645
OBSERVER_SAMPLING_UM = 0.5


@dataclass(frozen=True)
class BBBC039Record:
    """One source image with its untouched official split."""

    image_id: str
    split: str
    image_path: Path
    mask_path: Path
    excluded: bool


@dataclass(frozen=True)
class ObjectNormalization:
    """Training-derived global object-intensity transform."""

    lower: float
    upper: float

    def apply(self, raw: np.ndarray) -> np.ndarray:
        if self.upper <= self.lower:
            raise ValueError("Object-normalization upper bound must exceed lower bound")
        return np.clip((raw.astype(np.float32) - self.lower) / (self.upper - self.lower), 0, 1)


@dataclass(frozen=True)
class PreparedPatch:
    """Biological object plus static observer-resolution supervision."""

    image_id: str
    split: str
    origin_yx: tuple[int, int]
    object_image: np.ndarray
    instance_labels: np.ndarray
    centers_yx: np.ndarray
    valid_objects: np.ndarray


def _excluded_ids(path: Path = MANIFEST_ROOT / "bbbc039-exclusions.json") -> set[str]:
    payload = json.loads(path.read_text())
    return {
        image_id
        for split_values in payload["excluded_image_ids"].values()
        for image_id in split_values
    }


def records(include_excluded: bool = False) -> list[BBBC039Record]:
    """Load paths while preserving official split membership."""
    exclusions = _excluded_ids()
    masks = {
        path.stem: path
        for path in (BBBC039_ROOT / "masks" / "masks").glob("*.png")
        if not path.name.startswith("._")
    }
    result = []
    for image_path, split in discover_bbbc039():
        image_id = image_path.stem
        if image_id not in masks:
            raise FileNotFoundError(f"Missing BBBC039 mask for {image_id}")
        excluded = image_id in exclusions
        if include_excluded or not excluded:
            result.append(
                BBBC039Record(
                    image_id=image_id,
                    split=split,
                    image_path=image_path,
                    mask_path=masks[image_id],
                    excluded=excluded,
                )
            )
    return result


def records_for_split(split: str) -> list[BBBC039Record]:
    """Return non-overlapping records for one official split."""
    if split not in {"training", "validation", "test"}:
        raise ValueError(f"Unknown BBBC039 split: {split}")
    return [record for record in records() if record.split == split]


def load_raw_image(path: Path) -> np.ndarray:
    """Load the single-channel 16-bit microscope texture as float32."""
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2:
        raise ValueError(f"Expected 2D BBBC039 image at {path}, got {value.shape}")
    return np.asarray(value, dtype=np.float32)


def decode_instance_mask(path: Path) -> np.ndarray:
    """Follow the official BBBC039 decoder: channel zero then components."""
    with Image.open(path) as image:
        encoded = np.asarray(image)
    if encoded.ndim != 3 or encoded.shape[-1] < 1:
        raise ValueError(f"Expected color BBBC039 mask at {path}, got {encoded.shape}")
    foreground = encoded[..., 0] > 0
    return connected_components(foreground).astype(np.int32)


def split_patch(value: np.ndarray, origin_yx: tuple[int, int]) -> np.ndarray:
    """Extract one locked 256×256 crop."""
    top, left = origin_yx
    patch = value[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
    if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Patch {origin_yx} is outside image shape {value.shape}")
    return patch


def patch_origins(split: str) -> tuple[tuple[int, int], ...]:
    if split == "training":
        return TRAIN_PATCH_ORIGINS
    if split == "validation":
        return VALIDATION_PATCH_ORIGINS
    raise ValueError("Optimization patches are defined only for training and validation")


def _contained_object_ids(labels: np.ndarray, margin: int) -> list[int]:
    height, width = labels.shape
    keep = []
    for object_id in np.unique(labels):
        if object_id == 0:
            continue
        yy, xx = np.nonzero(labels == object_id)
        if (
            yy.min() >= margin
            and yy.max() < height - margin
            and xx.min() >= margin
            and xx.max() < width - margin
        ):
            keep.append(int(object_id))
    return keep


def _keyed_object_subset(object_ids: list[int], key: str, maximum: int) -> list[int]:
    if len(object_ids) <= maximum:
        return sorted(object_ids)
    ranked = sorted(
        object_ids,
        key=lambda object_id: hashlib.sha256(f"{key}:{object_id}".encode()).digest(),
    )
    return sorted(ranked[:maximum])


def select_patch_instances(
    labels: np.ndarray,
    key: str,
    valid_margin: int = 48,
    maximum_instances: int = 32,
) -> np.ndarray:
    """Keep fully valid objects and deterministically cap their count."""
    selected = _keyed_object_subset(
        _contained_object_ids(labels, valid_margin), key, maximum_instances
    )
    result = np.zeros(labels.shape, dtype=np.int32)
    for new_id, old_id in enumerate(selected, start=1):
        result[labels == old_id] = new_id
    return result


def observer_size(shape: tuple[int, int]) -> tuple[int, int]:
    """Return fixed 0.645→0.5 µm sampling dimensions."""
    scale = BIOLOGICAL_SAMPLING_UM / OBSERVER_SAMPLING_UM
    return tuple(int(np.floor(length * scale + 0.5)) for length in shape)


def resample_labels_for_observer(labels: np.ndarray) -> np.ndarray:
    """Nearest-neighbor label resampling inside the observer contract."""
    source = torch.from_numpy(labels.astype(np.float32))[None, None]
    resized = functional.interpolate(source, size=observer_size(labels.shape), mode="nearest")
    return resized[0, 0].to(torch.int32).numpy()


def padded_centers(
    labels: np.ndarray, maximum_instances: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed ground-truth centers and an object-validity mask."""
    centers = np.full((maximum_instances, 2), -1.0, dtype=np.float32)
    valid = np.zeros((maximum_instances,), dtype=bool)
    object_ids = [int(value) for value in np.unique(labels) if value != 0]
    if len(object_ids) > maximum_instances:
        raise ValueError(f"Found {len(object_ids)} objects after a {maximum_instances}-object cap")
    for index, object_id in enumerate(object_ids):
        yy, xx = np.nonzero(labels == object_id)
        if len(yy) == 0:
            continue
        centers[index] = (float(yy.mean()), float(xx.mean()))
        valid[index] = True
    return centers, valid


def prepare_patch(
    record: BBBC039Record,
    origin_yx: tuple[int, int],
    normalization: ObjectNormalization,
    valid_margin: int = 48,
) -> PreparedPatch:
    """Create one deterministic optics/observer training item."""
    raw_patch = split_patch(load_raw_image(record.image_path), origin_yx)
    mask_patch = split_patch(decode_instance_mask(record.mask_path), origin_yx)
    key = f"{record.image_id}:{origin_yx[0]}:{origin_yx[1]}"
    selected = select_patch_instances(mask_patch, key=key, valid_margin=valid_margin)
    observer_labels = resample_labels_for_observer(selected)
    centers, valid = padded_centers(observer_labels)
    return PreparedPatch(
        image_id=record.image_id,
        split=record.split,
        origin_yx=origin_yx,
        object_image=normalization.apply(raw_patch),
        instance_labels=observer_labels,
        centers_yx=centers,
        valid_objects=valid,
    )


def global_training_percentiles(
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
) -> ObjectNormalization:
    """Compute exact nearest-rank percentiles using a uint16 histogram."""
    histogram = np.zeros(65536, dtype=np.int64)
    total = 0
    for record in records_for_split("training"):
        raw = load_raw_image(record.image_path).astype(np.uint16)
        histogram += np.bincount(raw.reshape(-1), minlength=len(histogram))
        total += raw.size
    cumulative = np.cumsum(histogram)

    def percentile(percent: float) -> float:
        target = int(np.floor((percent / 100.0) * (total - 1))) + 1
        return float(np.searchsorted(cumulative, target, side="left"))

    return ObjectNormalization(percentile(lower_percentile), percentile(upper_percentile))
