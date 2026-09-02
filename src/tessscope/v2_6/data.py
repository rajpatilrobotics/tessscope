"""Frozen training-only patch materialization for the v2.6 two-mask route."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tessscope.v2.data.bbbc006 import PROJECT_ROOT, records_for_split
from tessscope.v2.data.patches import PreparedV2Patch, patch_origins, prepare_patch
from tessscope.v2.data.preprocess import load_global_normalization
from tessscope.v2_3.closed_loop import ClosedLoopBatch, materialize_closed_loop_batch

PARTITIONS = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "v2_6"
    / "training-development-confirmation.json"
)
QUERY_SEED = "tessscope-v2.6-two-mask-patches-2610"
PARTITION_NAMES = ("optimization", "development", "confirmation")
NOISE_SEEDS = (2610, 2613, 2616, 2619)


def _rank(field_id: str, origin_yx: tuple[int, int]) -> bytes:
    payload = f"{QUERY_SEED}:{field_id}:{origin_yx[0]}:{origin_yx[1]}"
    return hashlib.sha256(payload.encode()).digest()


def collect_two_mask_patches(
    partition: str,
    count: int = 12,
) -> list[PreparedV2Patch]:
    """Collect one deterministic valid patch per frozen training-only well."""
    if partition not in PARTITION_NAMES:
        raise ValueError(f"Unknown v2.6 training-only partition: {partition}")
    if count <= 0:
        raise ValueError("Patch count must be positive")
    frozen = json.loads(PARTITIONS.read_text())
    if frozen["status"] != "frozen_before_v2_6_diagnostic_or_optimization_metrics":
        raise ValueError("The v2.6 training-only partitions are not frozen")
    allowed_wells = set(frozen["wells"][partition])
    normalization = load_global_normalization()
    ranked = sorted(
        (
            (record, origin)
            for record in records_for_split("training")
            if record.well in allowed_wells
            for origin in patch_origins("training")
        ),
        key=lambda item: _rank(item[0].field_id, item[1]),
    )
    selected: list[PreparedV2Patch] = []
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
    raise ValueError(f"Could not collect {count} valid {partition} patches")


def materialize_two_mask_batches(partition: str) -> tuple[ClosedLoopBatch, ...]:
    """Materialize the frozen four batches and four matched noise seeds."""
    patches = collect_two_mask_patches(partition)
    return tuple(
        materialize_closed_loop_batch(
            patches[index : index + 3],
            noise_seed=NOISE_SEEDS[index // 3],
        )
        for index in range(0, len(patches), 3)
    )
