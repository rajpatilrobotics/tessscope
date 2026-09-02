"""Training-only patch selection and exposure-policy helpers for v2.6."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tessscope.v2.data.bbbc006 import PROJECT_ROOT, records_for_split
from tessscope.v2.data.patches import PreparedV2Patch, patch_origins, prepare_patch
from tessscope.v2.data.preprocess import load_global_normalization

PARTITIONS = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "v2_6"
    / "training-development-confirmation.json"
)
QUERY_SEED = "tessscope-v2.6-exposure-patches-2600"
ROW_BLOCKS = ("abcd", "efgh", "ijkl", "mnop")


def exposure_photons(total: float, first_fraction: float) -> tuple[float, float]:
    """Return first/second budgets while enforcing exact photon conservation."""
    if total <= 0.0 or not 0.0 < first_fraction < 1.0:
        raise ValueError("Exposure budget and fraction must be positive and interior")
    first = float(total * first_fraction)
    second = float(total - first)
    if not np.isclose(first + second, total):
        raise ValueError("Exposure allocation must conserve the total photon budget")
    return first, second


def _rank(record_field: str, origin: tuple[int, int]) -> bytes:
    payload = f"{QUERY_SEED}:{record_field}:{origin[0]}:{origin[1]}"
    return hashlib.sha256(payload.encode()).digest()


def collect_exposure_patches(partition: str) -> list[PreparedV2Patch]:
    """Select one valid training patch from each preregistered plate-row block."""
    if partition not in {"development", "confirmation"}:
        raise ValueError("Exposure audit uses only development or confirmation")
    frozen = json.loads(PARTITIONS.read_text())
    allowed_wells = set(frozen["wells"][partition])
    normalization = load_global_normalization()
    candidates = [
        (record, origin)
        for record in records_for_split("training")
        if record.well in allowed_wells
        for origin in patch_origins("training")
    ]
    selected = []
    for block in ROW_BLOCKS:
        ranked = sorted(
            (
                item
                for item in candidates
                if item[0].well[0] in block
            ),
            key=lambda item: _rank(item[0].field_id, item[1]),
        )
        for record, origin in ranked:
            patch = prepare_patch(record, origin, normalization)
            if np.count_nonzero(patch.valid_objects) > 0:
                selected.append(patch)
                break
        else:
            raise ValueError(f"No valid {partition} patch in row block {block}")
    return selected


def summarize_exposure_rows(rows: list[dict]) -> dict:
    """Summarize paired nonzero-depth exposure rows."""
    if not rows:
        raise ValueError("Exposure summary requires rows")
    return {
        "sample_count": len(rows),
        "well_count": len({row["well"] for row in rows}),
        "first_pq": float(np.mean([row["before_pq"] for row in rows])),
        "corrected_pq": float(np.mean([row["after_pq"] for row in rows])),
        "focus_mae_um": float(np.mean([abs(row["residual_depth_um"]) for row in rows])),
        "signed_direction_accuracy": float(
            np.mean(
                [
                    np.sign(row["predicted_depth_um"])
                    == np.sign(row["original_depth_um"])
                    for row in rows
                ]
            )
        ),
        "fraction_frames_improved": float(
            np.mean([row["after_pq"] > row["before_pq"] for row in rows])
        ),
        "first_photon_mean": float(np.mean([row["first_photon_mean"] for row in rows])),
        "second_photon_mean": float(np.mean([row["second_photon_mean"] for row in rows])),
    }


def select_development_fraction(rows: list[dict]) -> dict:
    """Apply the preregistered per-system development selection rule."""
    by_fraction: dict[float, list[dict]] = {}
    for row in rows:
        by_fraction.setdefault(float(row["first_fraction"]), []).append(row)
    summaries = {
        fraction: summarize_exposure_rows(values)
        for fraction, values in sorted(by_fraction.items())
    }
    half_first = summaries[0.5]["first_pq"]
    eligible = [
        (fraction, summary)
        for fraction, summary in summaries.items()
        if summary["first_pq"] >= half_first - 0.005
    ]
    fraction, summary = min(
        eligible,
        key=lambda item: (
            -item[1]["corrected_pq"],
            item[1]["focus_mae_um"],
            -item[1]["fraction_frames_improved"],
            abs(item[0] - 0.5),
            item[0],
        ),
    )
    return {
        "selected_first_fraction": fraction,
        "selected_summary": summary,
        "half_split_first_pq": half_first,
        "all_summaries": {str(key): value for key, value in summaries.items()},
    }
