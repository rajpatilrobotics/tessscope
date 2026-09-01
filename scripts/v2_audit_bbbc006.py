"""Freeze training-only BBBC006 intensity calibration and label limitations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tessscope.v2.data.bbbc006 import (
    load_reference_labels,
    load_stack,
    records_for_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMBER_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-w1-members.json"
)
OUTPUT = (
    PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-training-calibration.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Training calibration already exists: {OUTPUT}")
    member_payload = json.loads(MEMBER_MANIFEST.read_text())
    if member_payload["status"] != "complete":
        raise SystemExit("Complete BBBC006 member manifest is required")
    sampled_values = []
    training_records = records_for_split("training")
    for completed, record in enumerate(training_records, start=1):
        stack = load_stack(record)
        sampled_values.append(stack[:, ::16, ::16].reshape(-1))
        if completed % 50 == 0 or completed == len(training_records):
            print(json.dumps({"training_fields_sampled": completed}), flush=True)
    sample = np.concatenate(sampled_values).astype(np.float64)
    percentiles = {
        str(value): float(np.percentile(sample, value))
        for value in (0.0, 0.1, 1.0, 50.0, 99.0, 99.9, 100.0)
    }
    offset = percentiles["0.1"]
    scale = percentiles["99.9"] - offset
    if scale <= 0:
        raise ValueError("Training-only intensity scale is not positive")
    nonnegative_normalized = np.maximum((sample - offset) / scale, 0.0)
    mean_nonnegative_normalized = float(np.mean(nonnegative_normalized))
    exposure_gain = 1.0 / mean_nonnegative_normalized

    label_audit = {}
    for split in ("training", "validation"):
        counts = []
        contiguous = []
        records = records_for_split(split)
        for record in records:
            labels = load_reference_labels(record)
            identifiers = np.unique(labels)
            identifiers = identifiers[identifiers > 0]
            counts.append(len(identifiers))
            contiguous.append(
                np.array_equal(identifiers, np.arange(1, len(identifiers) + 1))
            )
        label_audit[split] = {
            "field_count": len(records),
            "median_instance_count": float(np.median(counts)),
            "minimum_instance_count": int(np.min(counts)),
            "maximum_instance_count": int(np.max(counts)),
            "all_positive_ids_contiguous": bool(all(contiguous)),
        }

    payload = {
        "dataset": "BBBC006v1",
        "status": "complete_training_only",
        "source_member_manifest": str(MEMBER_MANIFEST.relative_to(PROJECT_ROOT)),
        "source_member_manifest_sha256": file_sha256(MEMBER_MANIFEST),
        "sampling": {
            "policy": "every 16th pixel in y/x from all 7 planes of all training fields",
            "sample_count": int(sample.size),
            "training_field_count": len(training_records),
        },
        "training_intensity_percentiles_uint16": percentiles,
        "normalization": {
            "policy": "single training-only p0.1-to-p99.9 affine transform; no clipping",
            "offset": offset,
            "scale": scale,
            "per_image_normalization": False,
        },
        "exposure": {
            "policy": "one global training-only gain targets mean rate 1 before boundary loss",
            "mean_nonnegative_normalized": mean_nonnegative_normalized,
            "positive_sample_fraction": float(np.mean(nonnegative_normalized > 0)),
            "gain": exposure_gain,
        },
        "reference_labels": {
            "source": "official BBBC006 automated CellProfiler z16 instance labels",
            "manual_ground_truth": False,
            "claim_limit": "automated reference labels may contain systematic segmentation errors",
            "audit": label_audit,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "normalization": payload["normalization"],
                "reference_label_audit": label_audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
