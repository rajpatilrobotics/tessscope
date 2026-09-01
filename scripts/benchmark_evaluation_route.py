"""Benchmark the full validation route before any test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tessscope.data.bbbc039 import records_for_split
from tessscope.evaluation.route import benchmark_validation_route
from tessscope.observer.calibration import load_frozen_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=int,
        default=3,
        help="number of decontaminated validation sources to benchmark",
    )
    parser.add_argument(
        "--test-window-seconds",
        type=float,
        default=3600.0,
        help="local time reserved for the eventual locked test run",
    )
    parser.add_argument("--designs", type=int, default=5)
    parser.add_argument("--observer-batch-size", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/gate6/evaluation-route-benchmark.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_records = records_for_split("validation")
    if not 1 <= args.sources <= len(validation_records):
        raise SystemExit(
            f"--sources must lie in 1..{len(validation_records)}, got {args.sources}"
        )
    report = benchmark_validation_route(
        validation_records[: args.sources],
        available_seconds=args.test_window_seconds,
        design_count=args.designs,
        maximum_batch_size=args.observer_batch_size,
        transform=load_frozen_transform(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Gate 6 failed: evaluation route or time reserve is insufficient")


if __name__ == "__main__":
    main()
