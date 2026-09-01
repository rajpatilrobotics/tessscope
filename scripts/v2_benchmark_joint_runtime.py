"""Benchmark one realistic exact three-Tesseract step before freezing budgets."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    collect_patches,
    joint_value_and_gradient,
    materialize_joint_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    parser.add_argument(
        "--output", default="artifacts/runs/v2/gate4/runtime-benchmark.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    services = [
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    ]
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    patches = collect_patches("training", 3, seed=47)
    batch = materialize_joint_batch(patches, noise_seed=47)
    point = np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03], dtype=np.float32)

    start = time.perf_counter()
    value, gradient, branches = joint_value_and_gradient(
        *services,
        point,
        batch,
        calibration,
        JointObjectiveWeights(),
    )
    elapsed = time.perf_counter() - start
    projected_steps = 120
    report = {
        "hardware": "MacBook Air M2 local runtime",
        "batch": {"support_objects": 2, "query_objects": 1, "depths": 7},
        "exact_step_seconds": elapsed,
        "projected_120_step_minutes": elapsed * projected_steps / 60.0,
        "maximum_resident_set_size_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "objective_value": value,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "branches": branches,
        "proposed_budget": {
            "pilot_steps_per_weight": 12,
            "promoted_steps_per_design": 60,
            "exact_joint_starts_initial": 2,
            "validation_batches": 4,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
