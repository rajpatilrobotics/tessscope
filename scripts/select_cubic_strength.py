"""Tune the classical cubic-mask RMS strength using validation data only."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from tessscope.data.bbbc039 import (
    BIOLOGICAL_SAMPLING_UM,
    OBSERVER_SAMPLING_UM,
    ObjectNormalization,
    patch_origins,
    prepare_patch,
    records_for_split,
)
from tessscope.evaluation.metrics import instance_metrics, valid_region_labels
from tessscope.evaluation.route import segment_sensor_batch
from tessscope.observer.calibration import load_frozen_transform
from tessscope.observer.instanseg import load_frozen_model
from tessscope.optics.model import (
    cubic_support_energy_fraction,
    simulate_cubic_sensor,
)

DEPTHS_UM = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
PRIMARY_DEPTHS_UM = {-6.0, -4.0, -2.0, 2.0, 4.0, 6.0}
COARSE_STRENGTHS = (0.0, 0.6, 1.2, 1.8, 2.4)


@dataclass(frozen=True)
class CubicScore:
    rms_strength_radians: float
    mean_off_focus_pq: float
    worst_depth_pq: float
    focus_pq: float
    patch_count: int


def prepared_validation_patches(record_limit: int | None = None):
    normalization = ObjectNormalization(125.0, 1642.0)
    records = records_for_split("validation")
    if record_limit is not None:
        records = records[:record_limit]
    return [
        prepare_patch(record, origin, normalization)
        for record in records
        for origin in patch_origins("validation")
    ]


def cubic_sensor(strength: float, object_image: np.ndarray) -> np.ndarray:
    sensor = simulate_cubic_sensor(
        jnp.asarray(strength, dtype=jnp.float32),
        jnp.asarray(object_image[None], dtype=jnp.float32),
        jnp.asarray(DEPTHS_UM),
    )
    return np.maximum(np.asarray(sensor.block_until_ready()[0]), 0.0)


def score_strength(
    strength: float,
    patches,
    model: torch.nn.Module,
    device: torch.device,
) -> CubicScore:
    transform = load_frozen_transform()
    margin = int(
        np.floor(48 * BIOLOGICAL_SAMPLING_UM / OBSERVER_SAMPLING_UM + 0.5)
    )
    rows: list[tuple[float, float]] = []
    for patch in patches:
        predictions = segment_sensor_batch(
            model,
            cubic_sensor(strength, patch.object_image),
            patch.instance_labels.shape,
            device=device,
            maximum_batch_size=4,
            transform=transform,
        )
        for depth, prediction in zip(DEPTHS_UM, predictions, strict=True):
            target, valid_prediction = valid_region_labels(
                patch.instance_labels, prediction, margin
            )
            pq = instance_metrics(target, valid_prediction).panoptic_quality
            rows.append((float(depth), pq))
    by_depth = {
        float(depth): float(np.mean([pq for row_depth, pq in rows if row_depth == depth]))
        for depth in DEPTHS_UM
    }
    primary = [by_depth[depth] for depth in sorted(PRIMARY_DEPTHS_UM)]
    return CubicScore(
        rms_strength_radians=strength,
        mean_off_focus_pq=float(np.mean(primary)),
        worst_depth_pq=float(min(primary)),
        focus_pq=by_depth[0.0],
        patch_count=len(patches),
    )


def best_score(scores: list[CubicScore]) -> CubicScore:
    return max(
        scores,
        key=lambda score: (
            score.mean_off_focus_pq,
            score.worst_depth_pq,
            -score.rms_strength_radians,
        ),
    )


def refinement_strengths(coarse_best: float) -> tuple[float, float, float]:
    values = np.clip(
        np.asarray([coarse_best - 0.3, coarse_best, coarse_best + 0.3]),
        0.0,
        2.4,
    )
    if len(np.unique(values)) < 3:
        values = np.linspace(
            max(0.0, coarse_best - 0.6), min(2.4, coarse_best + 0.6), 3
        )
    return tuple(float(value) for value in values)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    coarse_patches = prepared_validation_patches(record_limit=10)
    coarse_scores = [
        score_strength(strength, coarse_patches, model, device)
        for strength in COARSE_STRENGTHS
    ]
    refinements = refinement_strengths(best_score(coarse_scores).rms_strength_radians)
    all_patches = prepared_validation_patches()
    refinement_scores = [
        score_strength(strength, all_patches, model, device)
        for strength in refinements
    ]
    selected = best_score(refinement_scores)
    support_by_depth = {
        str(float(depth)): float(
            cubic_support_energy_fraction(
                jnp.asarray(selected.rms_strength_radians, dtype=jnp.float32),
                jnp.asarray(depth, dtype=jnp.float32),
            )
        )
        for depth in DEPTHS_UM
    }
    minimum_support = min(support_by_depth.values())
    report = {
        "gate": "gate7_cubic_strength_selection",
        "split": "validation",
        "device": str(device),
        "coarse_record_count": 10,
        "coarse_strengths_rms_radians": list(COARSE_STRENGTHS),
        "coarse_scores": [asdict(score) for score in coarse_scores],
        "refinement_strengths_rms_radians": list(refinements),
        "refinement_scores": [asdict(score) for score in refinement_scores],
        "selection_rule": "mean off-focus PQ, then worst-depth PQ, then lower RMS",
        "selected": asdict(selected),
        "support_energy_fraction_by_depth": support_by_depth,
        "minimum_support_energy_fraction": minimum_support,
        "minimum_required_support_energy_fraction": 0.995,
        "passed": bool(minimum_support >= 0.995),
    }
    output = Path("artifacts/runs/gate7/cubic-strength-selection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Selected cubic mask requires the 128-pixel support fallback")


if __name__ == "__main__":
    main()
