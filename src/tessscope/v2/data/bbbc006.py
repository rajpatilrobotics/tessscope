"""Frozen BBBC006 v1 Hoechst z-stack records and well-level split hygiene."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / "data" / "external" / "BBBC006" / "v1"
IMAGE_ROOT = DATA_ROOT / "images"
LABEL_ROOT = DATA_ROOT / "labels"
SPLIT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-splits.json"

Z_PLANES = tuple(range(13, 20))
DEPTHS_UM = tuple(float(2 * (plane - 16)) for plane in Z_PLANES)
SITES = (1, 2)
WELL_ROWS = tuple("abcdefghijklmnop")
WELL_COLUMNS = tuple(range(1, 25))
SPLIT_SEED = "tessscope-v2-bbbc006-well-split-20260901"

IMAGE_PATTERN = re.compile(
    r"^mcf-z-stacks-03212011_(?P<well>[a-p]\d{2})_s(?P<site>[12])_"
    r"w(?P<channel>[12])(?P<uuid>[0-9a-f-]+)\.tif$",
    re.IGNORECASE,
)
LABEL_PATTERN = re.compile(
    r"^mcf-z-stacks-03212011_(?P<well>[a-p]\d{2})_s(?P<site>[12])\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BBBC006ImageName:
    """Metadata encoded in one official BBBC006 image filename."""

    well: str
    site: int
    channel: int
    uuid: str


@dataclass(frozen=True)
class BBBC006StackRecord:
    """One well/site z-stack and its z16 automated reference labels."""

    well: str
    site: int
    split: str
    image_paths: tuple[Path, ...]
    label_path: Path

    @property
    def field_id(self) -> str:
        return f"{self.well}_s{self.site}"


def parse_image_filename(name: str) -> BBBC006ImageName:
    """Parse well, site, channel, and UUID from an official image member name."""
    match = IMAGE_PATTERN.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"Not an official BBBC006 image filename: {name}")
    return BBBC006ImageName(
        well=match.group("well").lower(),
        site=int(match.group("site")),
        channel=int(match.group("channel")),
        uuid=match.group("uuid").lower(),
    )


def parse_label_filename(name: str) -> tuple[str, int]:
    """Parse one official automated z16 reference-label filename."""
    match = LABEL_PATTERN.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"Not an official BBBC006 label filename: {name}")
    return match.group("well").lower(), int(match.group("site"))


def all_wells() -> tuple[str, ...]:
    """Return the complete 16×24 well plate in row-major order."""
    return tuple(f"{row}{column:02d}" for row in WELL_ROWS for column in WELL_COLUMNS)


def build_well_splits() -> dict[str, tuple[str, ...]]:
    """Stratify every plate row into 18/3/3 train/validation/test wells."""
    result: dict[str, list[str]] = {"training": [], "validation": [], "test": []}
    for row in WELL_ROWS:
        row_wells = [f"{row}{column:02d}" for column in WELL_COLUMNS]
        ordered = sorted(
            row_wells,
            key=lambda well: hashlib.sha256(f"{SPLIT_SEED}:{well}".encode()).digest(),
        )
        result["training"].extend(ordered[:18])
        result["validation"].extend(ordered[18:21])
        result["test"].extend(ordered[21:24])
    return {split: tuple(sorted(wells)) for split, wells in result.items()}


def validate_well_splits(splits: dict[str, tuple[str, ...] | list[str]]) -> None:
    """Reject overlap, missing wells, wrong counts, or row imbalance."""
    if set(splits) != {"training", "validation", "test"}:
        raise ValueError("BBBC006 splits must be training, validation, and test")
    expected_counts = {"training": 288, "validation": 48, "test": 48}
    observed: set[str] = set()
    for split, expected_count in expected_counts.items():
        wells = list(splits[split])
        if len(wells) != expected_count or len(set(wells)) != expected_count:
            raise ValueError(f"Expected {expected_count} unique {split} wells")
        if observed.intersection(wells):
            raise ValueError("BBBC006 well splits overlap")
        observed.update(wells)
        per_row = {row: sum(well.startswith(row) for well in wells) for row in WELL_ROWS}
        expected_per_row = 18 if split == "training" else 3
        if set(per_row.values()) != {expected_per_row}:
            raise ValueError(f"BBBC006 {split} split is not row-stratified: {per_row}")
    if observed != set(all_wells()):
        raise ValueError("BBBC006 split manifest does not cover the complete plate")


def load_split_manifest(path: Path = SPLIT_MANIFEST) -> dict[str, tuple[str, ...]]:
    """Load the immutable pre-test well split."""
    payload = json.loads(path.read_text())
    if payload["status"] != "frozen_before_bbbc006_label_or_test_metric_access":
        raise ValueError("BBBC006 well split is not frozen")
    splits = {key: tuple(value) for key, value in payload["wells"].items()}
    validate_well_splits(splits)
    return splits


def split_for_well(well: str) -> str:
    """Return the frozen split for one normalized well identifier."""
    normalized = well.lower()
    matches = [split for split, wells in load_split_manifest().items() if normalized in wells]
    if len(matches) != 1:
        raise ValueError(f"Well {well} occurs in {len(matches)} frozen splits")
    return matches[0]


def records_for_split(
    split: str,
    *,
    allow_test: bool = False,
    require_files: bool = True,
) -> list[BBBC006StackRecord]:
    """Return complete field records while guarding accidental test access."""
    splits = load_split_manifest()
    if split not in splits:
        raise ValueError(f"Unknown BBBC006 split: {split}")
    if split == "test" and not allow_test:
        raise PermissionError("Test access requires allow_test=True in the locked evaluator")
    records: list[BBBC006StackRecord] = []
    for well in splits[split]:
        for site in SITES:
            paths = tuple(
                IMAGE_ROOT / f"z_{plane:02d}" / f"{well}_s{site}_w1.tif"
                for plane in Z_PLANES
            )
            label_path = LABEL_ROOT / f"{well}_s{site}.png"
            if require_files:
                missing = [path for path in (*paths, label_path) if not path.exists()]
                if missing:
                    raise FileNotFoundError(f"Missing BBBC006 field files: {missing[:3]}")
            records.append(
                BBBC006StackRecord(
                    well=well,
                    site=site,
                    split=split,
                    image_paths=paths,
                    label_path=label_path,
                )
            )
    return records


def load_stack(record: BBBC006StackRecord) -> np.ndarray:
    """Load seven 16-bit Hoechst planes without per-image normalization."""
    planes = []
    for path in record.image_paths:
        with Image.open(path) as image:
            value = np.asarray(image)
        if value.shape != (520, 696) or value.dtype != np.uint16:
            raise ValueError(f"Unexpected BBBC006 image at {path}: {value.shape} {value.dtype}")
        planes.append(value)
    return np.stack(planes)


def load_reference_labels(record: BBBC006StackRecord) -> np.ndarray:
    """Load automated CellProfiler z16 instance labels without calling them manual truth."""
    with Image.open(record.label_path) as image:
        labels = np.asarray(image)
    if labels.shape != (520, 696) or labels.ndim != 2:
        raise ValueError(f"Unexpected BBBC006 labels at {record.label_path}: {labels.shape}")
    return labels.astype(np.int32, copy=False)
