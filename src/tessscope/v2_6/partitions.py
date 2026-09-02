"""Deterministic training-only well partitions for v2.6."""

from __future__ import annotations

import hashlib
from collections import defaultdict

SEED = "tessscope-v2.6-training-dev-confirmation-20260902"


def rank_key(well: str) -> bytes:
    """Return the frozen deterministic within-row ranking key."""
    return hashlib.sha256(f"{SEED}:{well}".encode()).digest()


def build_partitions(training_wells: list[str]) -> dict[str, list[str]]:
    """Split every 18-well plate row into 12/3/3 disjoint groups."""
    by_row: dict[str, list[str]] = defaultdict(list)
    for well in training_wells:
        by_row[well[0]].append(well)
    if sorted(by_row) != list("abcdefghijklmnop"):
        raise ValueError("Expected all 16 BBBC006 plate rows")
    partitions = {"optimization": [], "development": [], "confirmation": []}
    for row in sorted(by_row):
        ranked = sorted(by_row[row], key=rank_key)
        if len(ranked) != 18:
            raise ValueError(f"Expected 18 training wells in row {row}, got {len(ranked)}")
        partitions["optimization"].extend(ranked[:12])
        partitions["development"].extend(ranked[12:15])
        partitions["confirmation"].extend(ranked[15:])
    return partitions
