"""Compare an isolated v2.4 hard-validation rerun with the frozen evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)
EXACT_NAME = "v2_3-exact-balanced-segmentation_only-step-14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="isolated reproduction report")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def metrics(document: dict) -> dict[str, float]:
    summary = document["summaries"][EXACT_NAME]
    stage = document["stage_correction"][EXACT_NAME]
    evidence = document["paired_evidence"][EXACT_NAME]
    return {
        "first_hard_pq": summary["hard_dense_off_focus_pq"],
        "focus_mae_um": summary["mae_um"],
        "signed_direction_accuracy": summary["signed_direction_accuracy"],
        "corrected_hard_pq": stage["hard_off_focus_pq_after"],
        "improved_fraction": stage["fraction_hard_frames_improved"],
        "exact_vs_stopped_mean": evidence["corrected_vs_stopped"]["mean_difference"],
        "exact_vs_stopped_ci_lower": evidence["corrected_vs_stopped"]["ci_lower_95"],
        "exact_vs_stopped_ci_upper": evidence["corrected_vs_stopped"]["ci_upper_95"],
        "exact_vs_piecewise_mean": evidence["corrected_vs_piecewise"]["mean_difference"],
        "exact_vs_piecewise_ci_lower": evidence["corrected_vs_piecewise"]["ci_lower_95"],
        "exact_vs_piecewise_ci_upper": evidence["corrected_vs_piecewise"]["ci_upper_95"],
    }


def main() -> None:
    args = parse_args()
    if args.tolerance <= 0:
        raise SystemExit("--tolerance must be positive")
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path = report_path.resolve()
    if not report_path.is_relative_to(PROJECT_ROOT):
        raise SystemExit("Reproduction report must stay inside the project directory")
    candidate = json.loads(report_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    if candidate.get("test_accessed") is not False:
        raise SystemExit("Reproduction report does not preserve the sealed-test boundary")
    candidate_metrics = metrics(candidate)
    reference_metrics = metrics(reference)
    differences = {
        name: abs(candidate_metrics[name] - reference_metrics[name])
        for name in reference_metrics
    }
    maximum = max(differences.values())
    payload = {
        "status": "passed" if maximum <= args.tolerance else "failed",
        "test_accessed": False,
        "tolerance": args.tolerance,
        "maximum_absolute_difference": maximum,
        "differences": differences,
    }
    print(json.dumps(payload, indent=2))
    if payload["status"] != "passed":
        raise SystemExit("Reproduced v2.4 evidence exceeds the allowed tolerance")


if __name__ == "__main__":
    main()
