"""Freeze the preregistered v2.4 soft checkpoint selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tessscope.v2_4.checkpoints import select_checkpoints

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT_ROOT / "configs" / "v2_4" / "contract.yaml"
POOL = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "preregistration"
    / "checkpoint-pool.json"
)
AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "checkpoint-audit.json"
)
OUTPUT = PROJECT_ROOT / "configs" / "v2_4" / "selected-checkpoints.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"V2.4 selection already exists: {OUTPUT}")
    audit = json.loads(AUDIT.read_text())
    selection = select_checkpoints(audit["rows"])
    if selection != audit["selection"]:
        raise SystemExit("Stored v2.4 selection does not reproduce from frozen rows")
    by_name = {row["name"]: row for row in audit["selected"]}
    selected = []
    for name in selection["selected_names"]:
        row = by_name[name]
        selected.append(
            {
                key: row[key]
                for key in (
                    "name",
                    "source_run",
                    "profile",
                    "start_name",
                    "step",
                    "parameter_sha256",
                    "parameters",
                    "objective",
                    "first_segmentation_loss",
                    "final_segmentation_loss",
                    "residual_mae_um",
                    "start_residual_mae_um",
                    "physical_coefficient_norm_radians",
                    "minimum_support_energy_fraction",
                )
            }
        )
    report = {
        "status": "soft_selection_frozen_before_stopped_or_hard_validation",
        "test_accessed": False,
        "hard_labels_accessed": False,
        "training_rerun_for_checkpoint_audit": False,
        "first_segmentation_maximum": audit["first_segmentation_maximum"],
        "pool_counts": {
            "checkpoint_records": audit["checkpoint_record_count"],
            "unique_parameter_hashes": audit["unique_parameter_hash_count"],
            "soft_eligible": len(selection["eligible_names"]),
            "soft_nondominated": len(selection["nondominated_names"]),
            "selected": len(selected),
        },
        "selection_rule": {
            "metrics": [
                "first_segmentation_loss",
                "final_segmentation_loss",
                "residual_mae_um",
            ],
            "maximum_promotions": 3,
            "one_per_source_run": True,
            "one_per_exact_parameter_hash": True,
            "tie_break": [
                "lowest_final_segmentation_loss",
                "lowest_residual_mae_um",
                "lowest_first_segmentation_loss",
                "deterministic_checkpoint_name",
            ],
        },
        "selected": selected,
        "evidence_sha256": {
            "contract": digest(CONTRACT),
            "checkpoint_pool": digest(POOL),
            "checkpoint_audit": digest(AUDIT),
        },
        "next_gate": "matched_stopped_stage_then_expanded_hard_validation",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "eligible": report["pool_counts"]["soft_eligible"],
                "nondominated": report["pool_counts"]["soft_nondominated"],
                "selected": [row["name"] for row in selected],
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
