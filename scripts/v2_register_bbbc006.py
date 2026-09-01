"""Register BBBC006 train/validation z-stacks to z16 and freeze translations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tessscope.v2.data.bbbc006 import DEPTHS_UM, load_stack, records_for_split
from tessscope.v2.data.registration import RegistrationConfig, register_translation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-registrations.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--upsample-factor", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "test" in args.splits:
        raise SystemExit("Registration tuning/audit cannot access the frozen test split")
    if not set(args.splits).issubset({"training", "validation"}):
        raise SystemExit("Registration splits must be training and/or validation")
    if args.output.exists():
        raise SystemExit(f"Registration output already exists: {args.output}")
    config = RegistrationConfig(
        downsample=args.downsample,
        upsample_factor=args.upsample_factor,
    )
    rows = []
    expected_records = 0
    for split in args.splits:
        records = records_for_split(split)
        expected_records += len(records)
        if args.max_records is not None:
            records = records[: args.max_records]
        for completed, record in enumerate(records, start=1):
            stack = load_stack(record)
            reference = stack[3]
            field_rows = []
            for index, (plane, depth_um) in enumerate(
                zip(range(13, 20), DEPTHS_UM, strict=True)
            ):
                if plane == 16:
                    result_row = {
                        "shift_yx_px": [0.0, 0.0],
                        "residual_shift_yx_px": [0.0, 0.0],
                        "correlation_before": 1.0,
                        "correlation_after": 1.0,
                        "status": "accepted",
                    }
                else:
                    try:
                        result = register_translation(reference, stack[index], config)
                    except ValueError as error:
                        result_row = {
                            "shift_yx_px": [0.0, 0.0],
                            "residual_shift_yx_px": None,
                            "correlation_before": None,
                            "correlation_after": None,
                            "status": "rejected_registration_failure",
                            "reason": str(error),
                        }
                    else:
                        correlation_accepted = (
                            result.correlation_before >= 0.20
                            and result.correlation_after
                            >= result.correlation_before - 0.02
                        )
                        result_row = {
                            "shift_yx_px": list(result.shift_yx_px),
                            "residual_shift_yx_px": list(result.residual_shift_yx_px),
                            "correlation_before": result.correlation_before,
                            "correlation_after": result.correlation_after,
                            "status": (
                                "accepted"
                                if correlation_accepted
                                else "rejected_low_content_or_correlation"
                            ),
                        }
                field_rows.append(
                    {
                        "split": split,
                        "well": record.well,
                        "site": record.site,
                        "field_id": record.field_id,
                        "z_plane": plane,
                        "depth_um": depth_um,
                        **result_row,
                    }
                )
            field_accepted = all(row["status"] == "accepted" for row in field_rows)
            for row in field_rows:
                row["field_accepted"] = field_accepted
            rows.extend(field_rows)
            if completed % 25 == 0 or completed == len(records):
                print(json.dumps({"split": split, "completed": completed}), flush=True)
    nonfocus = [
        row for row in rows if row["z_plane"] != 16 and row["field_accepted"]
    ]
    before = np.asarray([row["correlation_before"] for row in nonfocus])
    after = np.asarray([row["correlation_after"] for row in nonfocus])
    residual = np.asarray(
        [np.linalg.norm(row["residual_shift_yx_px"]) for row in nonfocus]
    )
    complete = args.max_records is None and set(args.splits) == {"training", "validation"}
    rejected_fields = sorted(
        {row["field_id"] for row in rows if not row["field_accepted"]}
    )
    processed_fields = len(rows) // 7
    accepted_by_split = {}
    for split in args.splits:
        split_fields = {
            row["field_id"] for row in rows if row["split"] == split
        }
        split_accepted = {
            row["field_id"]
            for row in rows
            if row["split"] == split and row["field_accepted"]
        }
        accepted_by_split[split] = {
            "accepted": len(split_accepted),
            "total": len(split_fields),
            "fraction": len(split_accepted) / len(split_fields),
        }
    audit = {
        "median_correlation_before": float(np.median(before)),
        "median_correlation_after": float(np.median(after)),
        "p95_residual_shift_px": float(np.quantile(residual, 0.95)),
        "fraction_correlation_improved": float(np.mean(after >= before)),
        "accepted_field_count": processed_fields - len(rejected_fields),
        "rejected_field_count": len(rejected_fields),
        "accepted_field_fraction": float(
            (processed_fields - len(rejected_fields)) / processed_fields
        ),
        "rejected_fields": rejected_fields,
        "accepted_by_split": accepted_by_split,
        "minimum_accepted_fraction_per_split": 0.90,
    }
    audit["passed"] = bool(
        min(row["fraction"] for row in accepted_by_split.values()) >= 0.90
        and audit["p95_residual_shift_px"] <= 1.0
        and audit["median_correlation_after"] >= audit["median_correlation_before"]
    )
    payload = {
        "dataset": "BBBC006v1",
        "reference_plane": 16,
        "method": "translation phase correlation on robust downsampled views",
        "field_rejection_policy": (
            "reject whole field if any plane exceeds 20 px, has correlation below 0.20, "
            "or worsens correlation by more than 0.02"
        ),
        "coverage_policy": (
            "at least 90% usable fields in each train/validation split; low-content fields "
            "cannot support reliable image-based registration or autofocus"
        ),
        "intensity_policy": "estimation views normalized; registered data retain global scale",
        "status": "complete_train_validation" if complete else "smoke_partial",
        "config": {
            "downsample": config.downsample,
            "upsample_factor": config.upsample_factor,
            "maximum_shift_px": config.maximum_shift_px,
        },
        "expected_record_count": expected_records,
        "processed_record_count": len(rows) // 7,
        "audit": audit,
        "registrations": rows,
    }
    output = (
        args.output
        if complete
        else PROJECT_ROOT / "artifacts" / "runs" / "v2" / "data" / "registration-smoke.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"], **audit}, indent=2))
    if complete and not audit["passed"]:
        raise SystemExit("Train/validation registration audit failed")


if __name__ == "__main__":
    main()
