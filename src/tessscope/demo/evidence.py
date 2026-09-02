"""Deterministic selection and traceability helpers for demo evidence."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

EXACT_NAME = "v2_3-exact-balanced-segmentation_only-step-14"
STOPPED_NAME = "v2_4-stopped-balanced-segmentation_only-step-14"
PIECEWISE_NAME = "v2_2-piecewise-028"
NONZERO_DEPTHS = (-6.0, -4.0, -2.0, 2.0, 4.0, 6.0)
SELECTION_SEED = "tessscope-demo-representative-v1"

FIELD_FEATURES = (
    "exact_before_pq_mean",
    "exact_after_pq_mean",
    "exact_improvement_fraction",
    "exact_residual_depth_mae_um",
    "exact_minus_stopped_after_pq_mean",
    "exact_minus_piecewise_after_pq_mean",
)
DEPTH_FEATURES = (
    "exact_before_pq",
    "exact_after_pq",
    "exact_after_minus_before_pq",
    "exact_residual_depth_absolute_um",
    "exact_minus_stopped_after_pq",
    "exact_minus_piecewise_after_pq",
)


def sha256_path(path: Path) -> str:
    """Return the SHA-256 of a file without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash one C-contiguous array with dtype and shape recorded separately."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed NumPy archive with fixed order and ZIP timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def tie_digest(identifier: str, *, seed: str = SELECTION_SEED) -> str:
    """Return the frozen deterministic tie breaker for one identifier."""
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def robust_rank(
    rows: Sequence[Mapping[str, float | str]],
    *,
    identifier_key: str,
    feature_names: Sequence[str],
    seed: str = SELECTION_SEED,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    """Rank rows by mean absolute robust-z distance from population medians."""
    if not rows:
        raise ValueError("Representative ranking requires at least one row")
    values = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Representative features must be finite")
    centers = np.median(values, axis=0)
    scales = 1.4826 * np.median(np.abs(values - centers), axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    distances = np.mean(np.abs((values - centers) / scales), axis=1)
    ranked = []
    for source, distance in zip(rows, distances, strict=True):
        identifier = str(source[identifier_key])
        ranked.append(
            {
                **source,
                "representative_distance": float(distance),
                "tie_sha256": tie_digest(identifier, seed=seed),
            }
        )
    ranked.sort(key=lambda row: (row["representative_distance"], row["tie_sha256"]))
    statistics = {
        name: {"median": float(center), "scaled_mad": float(scale)}
        for name, center, scale in zip(feature_names, centers, scales, strict=True)
    }
    return ranked, statistics


def rows_by_field_depth(rows: Iterable[dict], design: str) -> dict[tuple[str, float], dict]:
    """Index one design's corrected hard rows by frozen field and source depth."""
    selected = {}
    for row in rows:
        if row["design"] != design:
            continue
        key = (str(row["field_id"]), float(row["original_depth_um"]))
        if key in selected:
            raise ValueError(f"Duplicate corrected row for {design}: {key}")
        selected[key] = row
    return selected


def build_selection_rows(v2_4: dict, piecewise: dict) -> tuple[list[dict], dict]:
    """Build field aggregates and per-depth evidence for frozen systems."""
    exact = rows_by_field_depth(v2_4["corrected_rows"], EXACT_NAME)
    stopped = rows_by_field_depth(v2_4["corrected_rows"], STOPPED_NAME)
    piecewise_rows = rows_by_field_depth(piecewise["corrected_rows"], PIECEWISE_NAME)
    if set(exact) != set(stopped) or set(exact) != set(piecewise_rows):
        raise ValueError("Exact, stopped, and piecewise corrected rows do not align")

    field_ids = sorted({field_id for field_id, _ in exact})
    expected = {(field_id, depth) for field_id in field_ids for depth in NONZERO_DEPTHS}
    if set(exact) != expected:
        raise ValueError("Corrected evidence is not a complete 27-field by 6-depth grid")

    aggregates = []
    frame_rows: dict[str, list[dict]] = {}
    for field_id in field_ids:
        frames = []
        for depth in NONZERO_DEPTHS:
            key = (field_id, depth)
            exact_row = exact[key]
            stopped_row = stopped[key]
            piecewise_row = piecewise_rows[key]
            if not (
                exact_row["well"] == stopped_row["well"] == piecewise_row["well"]
            ):
                raise ValueError(f"Well mismatch for {key}")
            frames.append(
                {
                    "depth_id": f"{field_id}:{depth:+.1f}",
                    "well": str(exact_row["well"]),
                    "field_id": field_id,
                    "depth_um": depth,
                    "exact_before_pq": float(exact_row["before_pq"]),
                    "exact_after_pq": float(exact_row["after_pq"]),
                    "exact_after_minus_before_pq": float(
                        exact_row["after_pq"] - exact_row["before_pq"]
                    ),
                    "exact_residual_depth_um": float(exact_row["residual_depth_um"]),
                    "exact_residual_depth_absolute_um": abs(
                        float(exact_row["residual_depth_um"])
                    ),
                    "stopped_after_pq": float(stopped_row["after_pq"]),
                    "piecewise_after_pq": float(piecewise_row["after_pq"]),
                    "exact_minus_stopped_after_pq": float(
                        exact_row["after_pq"] - stopped_row["after_pq"]
                    ),
                    "exact_minus_piecewise_after_pq": float(
                        exact_row["after_pq"] - piecewise_row["after_pq"]
                    ),
                }
            )
        frame_rows[field_id] = frames
        aggregates.append(
            {
                "well": frames[0]["well"],
                "field_id": field_id,
                "exact_before_pq_mean": float(
                    np.mean([row["exact_before_pq"] for row in frames])
                ),
                "exact_after_pq_mean": float(
                    np.mean([row["exact_after_pq"] for row in frames])
                ),
                "exact_improvement_fraction": float(
                    np.mean(
                        [row["exact_after_pq"] > row["exact_before_pq"] for row in frames]
                    )
                ),
                "exact_residual_depth_mae_um": float(
                    np.mean([row["exact_residual_depth_absolute_um"] for row in frames])
                ),
                "exact_minus_stopped_after_pq_mean": float(
                    np.mean([row["exact_minus_stopped_after_pq"] for row in frames])
                ),
                "exact_minus_piecewise_after_pq_mean": float(
                    np.mean([row["exact_minus_piecewise_after_pq"] for row in frames])
                ),
            }
        )
    return aggregates, frame_rows


def select_representative(v2_4: dict, piecewise: dict) -> dict:
    """Apply the preregistered field and within-field depth selection."""
    fields, frames = build_selection_rows(v2_4, piecewise)
    ranked_fields, field_statistics = robust_rank(
        fields,
        identifier_key="field_id",
        feature_names=FIELD_FEATURES,
    )
    selected_field = str(ranked_fields[0]["field_id"])
    ranked_depths, depth_statistics = robust_rank(
        frames[selected_field],
        identifier_key="depth_id",
        feature_names=DEPTH_FEATURES,
    )
    return {
        "selection_rule": {
            "seed": SELECTION_SEED,
            "center": "median",
            "scale": "1.4826 * median absolute deviation; zero falls back to 1.0",
            "distance": "mean absolute robust z score",
            "tie_break": "ascending SHA-256 of seed:identifier",
            "field_features": list(FIELD_FEATURES),
            "display_depth_features": list(DEPTH_FEATURES),
        },
        "population": {
            "field_count": len(ranked_fields),
            "depths_um": list(NONZERO_DEPTHS),
            "frame_count": sum(len(rows) for rows in frames.values()),
        },
        "field_feature_statistics": field_statistics,
        "field_ranking": ranked_fields,
        "selected_field": ranked_fields[0],
        "selected_depth_feature_statistics": depth_statistics,
        "selected_field_depth_ranking": ranked_depths,
        "selected_display_depth": ranked_depths[0],
    }
