"""Select feasible full-training B7 checkpoints by held-out soft validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    collect_patches,
    materialize_joint_batch,
)
from tessscope.v2_1.optimization.constrained import is_promotion_eligible
from tessscope.v2_1.optimization.served import (
    averaged_forward,
    b7_coefficient_list,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_ROOT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2_1" / "optimization"
)
SEPARATE = OPTIMIZATION_ROOT / "b7-separate-baselines.json"
LOOSE = OPTIMIZATION_ROOT / "b7-constrained-full-training.json"
STRICT = OPTIMIZATION_ROOT / "b7-constrained-full-training-margin-0.004.json"
OUTPUT = OPTIMIZATION_ROOT / "b7-full-training-checkpoint-selection.json"
SLACK_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def checkpoint_rows(experiment: dict, family: str) -> list[dict]:
    """Return only full-training-feasible SLSQP callbacks from one run."""
    return [
        {
            "name": f"{family}-iteration-{row['iteration']}",
            "family": family,
            "iteration": row["iteration"],
            "parameters": row["parameters"],
            "phase_coefficients": b7_coefficient_list(
                np.asarray(row["parameters"], dtype=np.float32)
            ),
            "training": {
                "segmentation_loss": row["segmentation_loss"],
                "focus_mse": row["focus_mse"],
                "constraint_slack": row["constraint_slack"],
            },
        }
        for row in experiment["design"]["trace"]
        if row["constraint_slack"] >= -SLACK_TOLERANCE
    ]


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"Checkpoint-selection artifact already exists: {OUTPUT}")
    separate = load_json(SEPARATE)
    loose = load_json(LOOSE)
    strict = load_json(STRICT)
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    validation_patches = collect_patches("validation", 12, seed=53)
    validation_batches = [
        materialize_joint_batch(validation_patches[index : index + 3], noise_seed=None)
        for index in range(0, len(validation_patches), 3)
    ]
    validation_segmentation_limit = (
        separate["segmentation_only"]["validation"]["segmentation_loss"]
        + 0.008
    )
    candidates = [
        *checkpoint_rows(loose, "full-margin-0.008"),
        *checkpoint_rows(strict, "full-margin-0.004"),
    ]
    for index, candidate in enumerate(candidates, start=1):
        candidate["validation"] = averaged_forward(
            services,
            np.asarray(candidate["parameters"], dtype=np.float32),
            validation_batches,
            calibration,
        )
        candidate["eligible"] = is_promotion_eligible(
            candidate["training"]["constraint_slack"],
            candidate["validation"]["segmentation_loss"],
            validation_segmentation_limit,
            slack_tolerance=SLACK_TOLERANCE,
        )
        print(
            json.dumps(
                {
                    "checkpoint": index,
                    "name": candidate["name"],
                    "eligible": candidate["eligible"],
                    "validation_segmentation_loss": round(
                        candidate["validation"]["segmentation_loss"], 6
                    ),
                    "validation_focus_mse": round(
                        candidate["validation"]["focus_mse"], 6
                    ),
                }
            ),
            flush=True,
        )
    family_winners = []
    for family in sorted({row["family"] for row in candidates}):
        eligible_family = [
            row
            for row in candidates
            if row["family"] == family and row["eligible"]
        ]
        if eligible_family:
            family_winners.append(
                min(
                    eligible_family,
                    key=lambda row: (
                        row["validation"]["focus_mse"],
                        row["validation"]["segmentation_loss"],
                    ),
                )
            )
    ranked = sorted(
        family_winners,
        key=lambda row: (
            row["validation"]["focus_mse"],
            row["validation"]["segmentation_loss"],
        ),
    )
    selected = [row["name"] for row in ranked[:2]]
    report = {
        "status": "complete_full_training_checkpoint_selection_validation_only",
        "test_accessed": False,
        "method": (
            "validation-based early stopping over training-only SLSQP callbacks; "
            "one winner per predeclared epsilon-margin trajectory"
        ),
        "validation_segmentation_anchor": separate["segmentation_only"][
            "validation"
        ]["segmentation_loss"],
        "validation_segmentation_limit": validation_segmentation_limit,
        "minimum_training_constraint_slack": -SLACK_TOLERANCE,
        "validation_patch_ids": [
            (
                f"{patch.field_id}:{patch.origin_yx[0]}:"
                f"{patch.origin_yx[1]}"
            )
            for patch in validation_patches
        ],
        "candidates": candidates,
        "eligible_family_winners": [row["name"] for row in ranked],
        "selected_for_expanded_hard_validation": selected,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selected_for_expanded_hard_validation": selected,
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not selected:
        raise SystemExit(
            "No feasible full-training checkpoint passed soft validation; "
            "hard validation remains blocked"
        )


if __name__ == "__main__":
    main()
