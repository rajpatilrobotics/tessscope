"""Freeze well-grouped v2.6 optimization/development/confirmation partitions."""

from __future__ import annotations

import json
from pathlib import Path

from tessscope.v2_6.partitions import SEED, build_partitions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-splits.json"
OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "v2_6"
    / "training-development-confirmation.json"
)
def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"V2.6 partition manifest already exists: {OUTPUT}")
    source = json.loads(SOURCE.read_text())
    partitions = build_partitions(source["wells"]["training"])
    report = {
        "status": "frozen_before_v2_6_diagnostic_or_optimization_metrics",
        "dataset": "BBBC006v1",
        "source_split": "training",
        "split_unit": "well",
        "seed": SEED,
        "strategy": "within each plate row hash-order 12/3/3 wells",
        "well_counts": {name: len(wells) for name, wells in partitions.items()},
        "wells": partitions,
        "test_wells_accessed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "counts": report["well_counts"]}))


if __name__ == "__main__":
    main()
