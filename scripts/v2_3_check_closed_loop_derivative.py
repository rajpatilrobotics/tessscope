"""Gate the exact served v2.3 feedback-loop derivative and stage contribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration, collect_patches
from tessscope.v2_3.closed_loop import (
    OBJECTIVE_PROFILES,
    closed_loop_value,
    closed_loop_value_and_gradient,
    materialize_closed_loop_batch,
)
from tessscope.validation.derivatives import directional_derivative_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIECEWISE_SCREEN = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "screening"
    / "piecewise-soft-frontier.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_3"
    / "gates"
    / "closed-loop-derivative.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8407")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"V2.3 derivative artifact already exists: {OUTPUT}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    screen = json.loads(PIECEWISE_SCREEN.read_text())
    point = next(
        row for row in screen["points"] if row["name"] == "v2_2-piecewise-028"
    )
    parameters = np.asarray(point["parameters"], dtype=np.float32)
    patches = collect_patches("training", 3, seed=53)
    batch = materialize_closed_loop_batch(patches, noise_seed=None)
    calibration = V2SystemCalibration.load()
    weights = OBJECTIVE_PROFILES["balanced"]
    exact_value, exact_gradient, exact_branches = closed_loop_value_and_gradient(
        *services,
        parameters,
        batch,
        calibration,
        weights,
        gradient_mode="exact",
    )
    stopped_value, stopped_gradient, stopped_branches = (
        closed_loop_value_and_gradient(
            *services,
            parameters,
            batch,
            calibration,
            weights,
            gradient_mode="stop_stage",
        )
    )
    report = directional_derivative_report(
        lambda candidate: closed_loop_value(
            *services,
            candidate,
            batch,
            calibration,
            weights,
            gradient_mode="exact",
        )[0],
        parameters,
        exact_gradient,
        seed=83,
        minimum_stable_epsilons=2,
    )
    stage_gradient = exact_gradient - stopped_gradient
    stage_fraction = float(
        np.linalg.norm(stage_gradient) / max(np.linalg.norm(exact_gradient), 1e-12)
    )
    forward_parity = abs(exact_value - stopped_value)
    passed = bool(
        report["passed"]
        and forward_parity <= 1e-6
        and stage_fraction >= 0.01
    )
    report.update(
        {
            "component": "v2_3_full_served_closed_loop",
            "point_name": point["name"],
            "parameters": parameters.tolist(),
            "patch_ids": list(batch.patch_ids),
            "exact_value": exact_value,
            "stopped_stage_value": stopped_value,
            "forward_parity_absolute": forward_parity,
            "exact_gradient": exact_gradient.tolist(),
            "stopped_stage_gradient": stopped_gradient.tolist(),
            "stage_path_gradient": stage_gradient.tolist(),
            "stage_path_gradient_fraction": stage_fraction,
            "exact_branches": exact_branches,
            "stopped_stage_branches": stopped_branches,
            "thresholds_v2_3": {
                "maximum_relative_error": 0.01,
                "minimum_cosine": 0.99,
                "minimum_stage_path_gradient_fraction": 0.01,
                "maximum_forward_parity_absolute": 1e-6,
            },
            "passed_v2_3": passed,
            "test_accessed": False,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "relative_error": report["overall_median_relative_error"],
                "cosine": report["overall_cosine_agreement"],
                "stage_path_gradient_fraction": stage_fraction,
                "forward_parity_absolute": forward_parity,
                "passed": passed,
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit("V2.3 full feedback-loop derivative gate failed")


if __name__ == "__main__":
    main()
