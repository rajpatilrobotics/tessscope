"""Freeze the v2.4 candidate pool without evaluating validation checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_3"
    / "optimization"
    / "closed-loop-matrix.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "preregistration"
    / "checkpoint-pool.json"
)
SOURCE_SHA256 = "27ca34997f2ff574093c00fee2f06b6987ee902ce98801b1d3ae2a08cecebcaf"


def parameter_sha256(parameters: list[float]) -> str:
    """Hash the exact compact JSON representation frozen in the source artifact."""
    payload = json.dumps(parameters, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"V2.4 checkpoint pool already exists: {OUTPUT}")
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_digest != SOURCE_SHA256:
        raise SystemExit("Frozen v2.3 source artifact hash mismatch")
    source = json.loads(SOURCE.read_text())
    records = []
    controls_by_start = {}
    for run in source["exact_runs"]:
        if len(run["trace"]) != 30:
            raise SystemExit(f"Expected 30 checkpoints for {run['name']}")
        controls_by_start.setdefault(
            run["start_name"],
            {
                "name": f"v2_4-control-{run['start_name']}",
                "start_name": run["start_name"],
                "parameter_sha256": parameter_sha256(run["start_parameters"]),
                "parameters": run["start_parameters"],
                "selectable": False,
            },
        )
        for trace_row in run["trace"]:
            step = int(trace_row["step"])
            parameters = trace_row["parameters"]
            records.append(
                {
                    "name": f"{run['name']}-step-{step:02d}",
                    "source_run": run["name"],
                    "profile": run["profile"],
                    "start_name": run["start_name"],
                    "step": step,
                    "parameter_sha256": parameter_sha256(parameters),
                }
            )
    if len(records) != 270 or len({row["name"] for row in records}) != 270:
        raise SystemExit("Frozen pool must contain exactly 270 unique checkpoint records")
    report = {
        "status": "frozen_before_intermediate_validation",
        "test_accessed": False,
        "source_artifact": str(SOURCE.relative_to(PROJECT_ROOT)),
        "source_artifact_sha256": source_digest,
        "hash_encoding": "sha256(compact JSON float list, allow_nan=false)",
        "source_run_count": len(source["exact_runs"]),
        "checkpoint_records_per_run": 30,
        "checkpoint_record_count": len(records),
        "unique_parameter_hash_count": len({row["parameter_sha256"] for row in records}),
        "intermediate_validation_metrics_computed": False,
        "controls": list(controls_by_start.values()),
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "source_runs": report["source_run_count"],
                "checkpoint_records": report["checkpoint_record_count"],
                "unique_parameter_hashes": report["unique_parameter_hash_count"],
                "intermediate_validation_metrics_computed": False,
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
