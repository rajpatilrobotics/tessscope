"""Run the frozen v2.6 oracle and controller-headroom diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tessscope.v2_6.diagnostics import (
    controller_error_decomposition,
    controller_grid,
    interpolated_controller_rows,
    oracle_rows,
    paired_well_evidence,
    summarize_controller_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "configs" / "v2_6" / "source-manifest.json"
V2_4 = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)
V2_2 = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "validation"
    / "piecewise-hard-frontier.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "diagnostics"
    / "oracle-controller-audit.json"
)
CANDIDATE = "v2_3-exact-balanced-segmentation_only-step-14"
BASELINE = "v2_2-piecewise-028"


def verify_sources() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    for source in manifest["sources"].values():
        actual = hashlib.sha256((PROJECT_ROOT / source["path"]).read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {source['path']}")
    return manifest


def grid_report(rows: list[dict], design: str) -> list[dict]:
    report = []
    for calibration in controller_grid():
        corrected = interpolated_controller_rows(rows, design, calibration)
        report.append(
            {
                "gain": calibration.gain,
                "bias_um": calibration.bias_um,
                "cubic": calibration.cubic,
                "monotone": calibration.is_monotone(),
                "summary": summarize_controller_rows(corrected),
                "rows": corrected,
            }
        )
    return sorted(
        report,
        key=lambda row: (
            -row["summary"]["corrected_pq"],
            row["summary"]["focus_mae_um"],
            -row["summary"]["fraction_frames_improved"],
            row["gain"],
            row["bias_um"],
            row["cubic"],
        ),
    )


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"V2.6 oracle audit already exists: {OUTPUT}")
    manifest = verify_sources()
    v2_4 = json.loads(V2_4.read_text())
    v2_2 = json.loads(V2_2.read_text())
    candidate_actual = [
        row for row in v2_4["corrected_rows"] if row["design"] == CANDIDATE
    ]
    baseline_actual = [
        row for row in v2_2["corrected_rows"] if row["design"] == BASELINE
    ]
    candidate_oracle = oracle_rows(v2_4["rows"], CANDIDATE)
    baseline_oracle = oracle_rows(v2_2["rows"], BASELINE)
    candidate_grid = grid_report(v2_4["rows"], CANDIDATE)
    baseline_grid = grid_report(v2_2["rows"], BASELINE)
    best_candidate = candidate_grid[0]
    best_baseline = baseline_grid[0]
    matched_grid_evidence = paired_well_evidence(
        best_candidate["rows"], best_baseline["rows"]
    )
    oracle_evidence = paired_well_evidence(candidate_oracle, baseline_actual)
    current_evidence = paired_well_evidence(candidate_actual, baseline_actual)
    actual_vs_interpolated = paired_well_evidence(
        candidate_actual,
        next(
            row["rows"]
            for row in candidate_grid
            if row["gain"] == 1.0 and row["bias_um"] == 0.0 and row["cubic"] == 0.0
        ),
    )
    minimum_baseline_pq = (
        v2_2["stage_correction"][BASELINE]["hard_off_focus_pq_after"] + 0.005
    )
    oracle_summary = summarize_controller_rows(candidate_oracle)
    matched_summary = best_candidate["summary"]
    route_checks = {
        "oracle_corrected_pq_at_least_actual_baseline_plus_0_005": bool(
            oracle_summary["corrected_pq"] >= minimum_baseline_pq
        ),
        "matched_grid_gain_at_least_0_005": bool(
            matched_grid_evidence["mean_difference"] >= 0.005
        ),
        "matched_grid_bootstrap_lower_positive": bool(
            matched_grid_evidence["ci_lower_95"] > 0.0
        ),
        "matched_grid_positive_well_fraction_at_least_0_60": bool(
            matched_grid_evidence["positive_well_fraction"] >= 0.60
        ),
        "matched_grid_leave_one_out_always_positive": bool(
            matched_grid_evidence["minimum_leave_one_well_out_mean_difference"] > 0.0
        ),
        "candidate_grid_focus_mae_at_most_1_0": bool(
            matched_summary["focus_mae_um"] <= 1.0
        ),
    }
    report = {
        "status": "complete_historical_validation_diagnostic_only",
        "test_accessed": False,
        "historical_validation_reuse_limitation": (
            "The 27 hard-density validation wells were used in prior v2.2-v2.4 work. "
            "This audit estimates headroom only and cannot freeze a new controller."
        ),
        "source_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "frozen_sources": manifest["sources"],
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "minimum_corrected_pq_from_actual_baseline": minimum_baseline_pq,
        "current_candidate": {
            "summary": summarize_controller_rows(candidate_actual),
            "vs_actual_baseline": current_evidence,
        },
        "zero_residual_oracles": {
            "candidate": summarize_controller_rows(candidate_oracle),
            "baseline": summarize_controller_rows(baseline_oracle),
            "candidate_vs_actual_baseline": oracle_evidence,
        },
        "interpolation_audit": {
            "actual_minus_default_interpolated": actual_vs_interpolated,
        },
        "controller_error_decomposition": controller_error_decomposition(
            v2_4["rows"], CANDIDATE
        ),
        "controller_grid": {
            "evaluated_per_system": len(candidate_grid),
            "best_candidate": best_candidate,
            "best_baseline": best_baseline,
            "best_candidate_vs_best_baseline": matched_grid_evidence,
            "top_candidate_summaries": [
                {key: value for key, value in row.items() if key != "rows"}
                for row in candidate_grid[:10]
            ],
            "top_baseline_summaries": [
                {key: value for key, value in row.items() if key != "rows"}
                for row in baseline_grid[:10]
            ],
        },
        "route_checks_before_exposure_audit": route_checks,
        "controller_route_supported_before_exposure_audit": all(route_checks.values()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "oracle_corrected_pq": oracle_summary["corrected_pq"],
                "minimum_corrected_pq": minimum_baseline_pq,
                "best_candidate_grid": matched_summary,
                "best_baseline_grid": best_baseline["summary"],
                "matched_grid_gain": matched_grid_evidence["mean_difference"],
                "controller_route_supported_before_exposure": all(route_checks.values()),
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
