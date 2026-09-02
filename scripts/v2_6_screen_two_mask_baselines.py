"""Soft-screen the frozen 76-pair matched two-mask piecewise family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration
from tessscope.v2_5.evaluation import physical_diagnostics
from tessscope.v2_6.data import materialize_two_mask_batches
from tessscope.v2_6.selection import select_soft_rows, soft_eligible
from tessscope.v2_6.two_mask import (
    average_two_mask_forward,
    join_two_mask_parameters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT_ROOT / "configs" / "v2_6" / "two-mask-contract.yaml"
SOURCES = PROJECT_ROOT / "configs" / "v2_6" / "two-mask-sources.json"
FRONTIER = (
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
    / "v2_6"
    / "two-mask"
    / "matched-baseline-screen.json"
)
RESIDUAL_LIMIT = 0.055


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
        raise SystemExit(f"V2.6 matched baseline artifact already exists: {OUTPUT}")
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    development_batches = materialize_two_mask_batches("development")
    confirmation_batches = materialize_two_mask_batches("confirmation")
    sources = json.loads(SOURCES.read_text())
    frontier = json.loads(FRONTIER.read_text())
    by_name = {row["name"]: row for row in frontier["points"]}
    sensing_names = frontier["selected_for_hard_validation"]
    if len(sensing_names) != 38:
        raise ValueError("Frozen v2.2 sensing frontier must contain exactly 38 points")
    capture_masks = {
        "b7-segmentation-only": sources["parameters"]["b7_segmentation_only"],
        "v2_2-piecewise-028": sources["parameters"]["v2_2_piecewise_028"],
    }
    segmentation = np.asarray(
        sources["parameters"]["b7_segmentation_only"], dtype=np.float32
    )
    anchor_parameters = join_two_mask_parameters(segmentation, segmentation)
    anchors = {
        partition: average_two_mask_forward(
            services,
            anchor_parameters,
            batches,
            calibration,
        )
        for partition, batches in (
            ("development", development_batches),
            ("confirmation", confirmation_batches),
        )
    }
    physical_cache: dict[str, dict] = {}

    def physical(name: str, parameters: np.ndarray) -> dict:
        if name not in physical_cache:
            physical_cache[name] = physical_diagnostics(parameters, basis_size=7)
        return physical_cache[name]

    development_rows = []
    total = len(sensing_names) * len(capture_masks)
    index = 0
    for sensing_name in sensing_names:
        sensing = np.asarray(by_name[sensing_name]["parameters"], dtype=np.float32)
        for capture_name, capture_values in capture_masks.items():
            index += 1
            capture = np.asarray(capture_values, dtype=np.float32)
            name = f"v2_6-baseline-{sensing_name}--{capture_name}"
            metrics = average_two_mask_forward(
                services,
                join_two_mask_parameters(sensing, capture),
                development_batches,
                calibration,
            )
            pair_physical = {
                "sensing": physical(sensing_name, sensing),
                "capture": physical(capture_name, capture),
            }
            eligible = soft_eligible(
                metrics,
                pair_physical,
                first_limit=anchors["development"]["first_segmentation_loss"] + 0.008,
                residual_limit=RESIDUAL_LIMIT,
            )
            development_rows.append(
                {
                    "name": name,
                    "sensing_name": sensing_name,
                    "capture_name": capture_name,
                    "sensing_parameters": sensing.tolist(),
                    "capture_parameters": capture.tolist(),
                    "metrics": metrics,
                    "physical": pair_physical,
                    "eligible": eligible,
                }
            )
            print(
                json.dumps(
                    {
                        "pair": index,
                        "of": total,
                        "name": name,
                        "final": round(metrics["final_segmentation_loss"], 6),
                        "residual_mae_um": round(metrics["residual_mae_um"], 5),
                        "eligible": eligible,
                    }
                ),
                flush=True,
            )
    selected = select_soft_rows(development_rows, maximum=3)
    selected_by_name = {row["name"]: row for row in development_rows}
    confirmation_rows = []
    for name in selected:
        development = selected_by_name[name]
        metrics = average_two_mask_forward(
            services,
            join_two_mask_parameters(
                development["sensing_parameters"],
                development["capture_parameters"],
            ),
            confirmation_batches,
            calibration,
        )
        pair_physical = development["physical"]
        confirmation_rows.append(
            {
                **{key: value for key, value in development.items() if key != "metrics"},
                "development_metrics": development["metrics"],
                "metrics": metrics,
                "eligible": soft_eligible(
                    metrics,
                    pair_physical,
                    first_limit=(
                        anchors["confirmation"]["first_segmentation_loss"] + 0.008
                    ),
                    residual_limit=RESIDUAL_LIMIT,
                ),
            }
        )
    best_confirmation = select_soft_rows(confirmation_rows, maximum=1)
    result = {
        "status": (
            "complete_training_only"
            if best_confirmation
            else "complete_training_only_no_eligible_baseline"
        ),
        "test_accessed": False,
        "contract_sha256": _sha256(CONTRACT),
        "sources_sha256": _sha256(SOURCES),
        "frontier_sha256": _sha256(FRONTIER),
        "budget": {
            "sensing_masks": len(sensing_names),
            "capture_masks": len(capture_masks),
            "total_pairs": len(development_rows),
            "development_batches": len(development_batches),
            "maximum_confirmation_baselines": 3,
        },
        "residual_limit": RESIDUAL_LIMIT,
        "anchors": anchors,
        "development_first_limit": (
            anchors["development"]["first_segmentation_loss"] + 0.008
        ),
        "confirmation_first_limit": (
            anchors["confirmation"]["first_segmentation_loss"] + 0.008
        ),
        "development_rows": development_rows,
        "selected_for_confirmation": selected,
        "confirmation_rows": confirmation_rows,
        "best_confirmation_baseline_name": (
            best_confirmation[0] if best_confirmation else None
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "eligible_development": sum(
                    row["eligible"] for row in development_rows
                ),
                "selected_for_confirmation": selected,
                "best_confirmation_baseline_name": result[
                    "best_confirmation_baseline_name"
                ],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
