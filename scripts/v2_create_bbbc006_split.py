"""Freeze the row-stratified BBBC006 well split before any test metric access."""

from __future__ import annotations

import json

from tessscope.v2.data.bbbc006 import (
    SPLIT_MANIFEST,
    SPLIT_SEED,
    build_well_splits,
    validate_well_splits,
)


def main() -> None:
    if SPLIT_MANIFEST.exists():
        raise SystemExit("BBBC006 split manifest already exists; refusing to overwrite it")
    splits = build_well_splits()
    validate_well_splits(splits)
    payload = {
        "dataset": "BBBC006v1",
        "status": "frozen_before_bbbc006_label_or_test_metric_access",
        "split_unit": "well",
        "seed": SPLIT_SEED,
        "strategy": "within each of 16 plate rows, hash-order 18/3/3 wells",
        "well_counts": {split: len(wells) for split, wells in splits.items()},
        "field_counts": {split: 2 * len(wells) for split, wells in splits.items()},
        "wells": splits,
    }
    SPLIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(SPLIT_MANIFEST), **payload["well_counts"]}, indent=2))


if __name__ == "__main__":
    main()
