"""Gate joint, sensing-only, and capture-only derivatives for two sequential masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration
from tessscope.v2_6.data import materialize_two_mask_batches
from tessscope.v2_6.two_mask import (
    FINAL_WEIGHTS,
    join_two_mask_parameters,
    two_mask_value,
    two_mask_value_and_gradient,
)
from tessscope.validation.derivatives import directional_derivative_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT_ROOT / "configs" / "v2_6" / "two-mask-contract.yaml"
SOURCES = PROJECT_ROOT / "configs" / "v2_6" / "two-mask-sources.json"
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "gates"
    / "two-mask-derivative.json"
)
EPSILONS = (0.01, 0.001, 0.0001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8407")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"V2.6 two-mask derivative artifact already exists: {OUTPUT}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    sources = json.loads(SOURCES.read_text())
    sensing = np.asarray(
        sources["parameters"]["v2_2_piecewise_028"], dtype=np.float32
    )
    capture = np.asarray(
        sources["parameters"]["b7_segmentation_only"], dtype=np.float32
    )
    parameters = join_two_mask_parameters(sensing, capture)
    batch = materialize_two_mask_batches("optimization")[0]
    calibration = V2SystemCalibration.load()
    exact_value, exact_gradient, exact_terms = two_mask_value_and_gradient(
        *services,
        parameters,
        batch,
        calibration,
        FINAL_WEIGHTS,
        gradient_mode="exact",
    )
    stopped_value, stopped_gradient, stopped_terms = two_mask_value_and_gradient(
        *services,
        parameters,
        batch,
        calibration,
        FINAL_WEIGHTS,
        gradient_mode="stop_stage",
    )

    def value(candidate: np.ndarray) -> float:
        return two_mask_value(
            *services,
            candidate,
            batch,
            calibration,
            FINAL_WEIGHTS,
            gradient_mode="exact",
        )[0]

    reports = {
        "joint": directional_derivative_report(
            value,
            parameters,
            exact_gradient,
            seed=2610,
            epsilons=EPSILONS,
            minimum_stable_epsilons=2,
        ),
        "sensing_only": directional_derivative_report(
            lambda candidate: value(join_two_mask_parameters(candidate, capture)),
            sensing,
            exact_gradient[:7],
            seed=2613,
            epsilons=EPSILONS,
            minimum_stable_epsilons=2,
        ),
        "capture_only": directional_derivative_report(
            lambda candidate: value(join_two_mask_parameters(sensing, candidate)),
            capture,
            exact_gradient[7:],
            seed=2616,
            epsilons=EPSILONS,
            minimum_stable_epsilons=2,
        ),
    }
    stage_gradient = exact_gradient[:7] - stopped_gradient[:7]
    stage_fraction = float(
        np.linalg.norm(stage_gradient)
        / max(np.linalg.norm(exact_gradient[:7]), 1e-12)
    )
    forward_parity = abs(exact_value - stopped_value)
    passed = bool(
        all(report["passed"] for report in reports.values())
        and stage_fraction >= 0.01
        and forward_parity <= 1e-6
    )
    result = {
        "status": "complete" if passed else "failed",
        "component": "v2_6_two_mask_full_served_loop",
        "test_accessed": False,
        "contract_sha256": _sha256(CONTRACT),
        "sources_sha256": _sha256(SOURCES),
        "parameters": parameters.tolist(),
        "patch_ids": list(batch.patch_ids),
        "exact_value": exact_value,
        "stopped_stage_value": stopped_value,
        "forward_parity_absolute": forward_parity,
        "exact_gradient": exact_gradient.tolist(),
        "stopped_stage_gradient": stopped_gradient.tolist(),
        "sensing_stage_path_gradient": stage_gradient.tolist(),
        "stage_path_gradient_fraction": stage_fraction,
        "exact_terms": exact_terms,
        "stopped_stage_terms": stopped_terms,
        "direction_reports": reports,
        "thresholds": {
            "maximum_median_relative_error": 0.01,
            "minimum_cosine": 0.99,
            "minimum_stage_path_gradient_fraction": 0.01,
            "maximum_forward_parity_absolute": 1e-6,
        },
        "passed_v2_6": passed,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "direction_metrics": {
                    name: {
                        "relative_error": report[
                            "overall_median_relative_error"
                        ],
                        "cosine": report["overall_cosine_agreement"],
                        "passed": report["passed"],
                    }
                    for name, report in reports.items()
                },
                "stage_path_gradient_fraction": stage_fraction,
                "forward_parity_absolute": forward_parity,
                "passed": passed,
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit("V2.6 two-mask derivative gate failed")


if __name__ == "__main__":
    main()
