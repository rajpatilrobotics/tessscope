"""Validation-only global observer-transform calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tessscope.observer.loss import ObserverTransform

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TransformScore:
    """Validation score used to choose one frozen global transform."""

    name: str
    mean_off_focus_pq: float
    worst_depth_pq: float
    focus_pq: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class UnitIntervalHistogram:
    """Fixed-bin streaming quantiles for physically bounded sensor rates."""

    def __init__(self, bins: int = 65_536) -> None:
        if bins < 2:
            raise ValueError("Histogram requires at least two bins")
        self.bins = bins
        self.counts = np.zeros((bins,), dtype=np.int64)
        self.total = 0

    def update(self, values: np.ndarray) -> None:
        value = np.asarray(values, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError("Sensor calibration values must be finite")
        clipped = np.clip(value, 0.0, 1.0)
        indices = np.floor(clipped * (self.bins - 1) + 0.5).astype(np.int64)
        self.counts += np.bincount(indices.reshape(-1), minlength=self.bins)
        self.total += value.size

    def percentile(self, percent: float) -> float:
        if self.total == 0:
            raise ValueError("Cannot compute a percentile from an empty histogram")
        if not 0 <= percent <= 100:
            raise ValueError("Percentile must lie in [0, 100]")
        rank = int(np.floor((percent / 100.0) * (self.total - 1))) + 1
        index = int(np.searchsorted(np.cumsum(self.counts), rank, side="left"))
        return index / (self.bins - 1)


def transform_candidates(lower: float, upper: float) -> dict[str, ObserverTransform]:
    """Return the two predeclared global transforms with shared calibration."""
    if upper <= lower:
        raise ValueError("Global calibration upper value must exceed lower value")
    scale = upper - lower
    return {
        "global_affine": ObserverTransform(mode="affine", offset=lower, scale=scale),
        "global_asinh": ObserverTransform(mode="asinh", offset=lower, scale=scale),
    }


def load_frozen_transform(
    path: Path = PROJECT_ROOT / "configs" / "preprocessing.json",
) -> ObserverTransform:
    """Load the validation-frozen global transform from the tracked config."""
    payload = json.loads(path.read_text())["observer_transform"]
    if payload["status"] != "frozen_after_validation-only selection":
        raise ValueError("Observer transform has not been frozen")
    return ObserverTransform(
        mode=payload["mode"],
        offset=float(payload["offset"]),
        scale=float(payload["scale"]),
    )


def choose_transform(
    scores: list[TransformScore],
    *,
    standard_focus_pq: float,
    maximum_focus_drop: float = 0.03,
) -> TransformScore:
    """Use affine when it passes; otherwise require the asinh fallback to pass."""
    if {score.name for score in scores} != {"global_affine", "global_asinh"}:
        raise ValueError("Scores must contain exactly global_affine and global_asinh")
    if not 0 <= maximum_focus_drop <= 1:
        raise ValueError("Maximum focus drop must lie in [0, 1]")
    floor = standard_focus_pq - maximum_focus_drop
    by_name = {score.name: score for score in scores}
    if by_name["global_affine"].focus_pq >= floor:
        return by_name["global_affine"]
    if by_name["global_asinh"].focus_pq >= floor:
        return by_name["global_asinh"]
    raise ValueError("Neither global observer transform passes the focus-PQ gate")
