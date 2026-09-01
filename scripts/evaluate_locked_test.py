"""Run the one-shot frozen deterministic and Poisson test evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from tessscope.data.bbbc039 import ObjectNormalization, observer_size, records_for_split
from tessscope.evaluation.metrics import (
    grouped_bootstrap_difference,
    instance_metrics,
    resample_labels,
    valid_region_labels,
)
from tessscope.evaluation.route import (
    deterministic_sensor,
    full_source_inputs,
    poisson_sensor_batch,
    segment_sensor_batch,
)
from tessscope.observer.calibration import load_frozen_transform
from tessscope.observer.instanseg import load_frozen_model
from tessscope.optics.model import simulate_cubic_sensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "runs" / "gate9"
DEPTHS_UM = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
PRIMARY_DEPTHS_UM = {-6.0, -4.0, -2.0, 2.0, 4.0, 6.0}


def verify_freeze() -> None:
    """Fail before test access if any frozen input differs from its recorded hash."""
    freeze = json.loads((PROJECT_ROOT / "configs" / "test-freeze.json").read_text())
    if freeze["status"] != "frozen_before_first_test_evaluation":
        raise ValueError("Test configuration is not frozen")
    mismatches = []
    for relative_path, expected in freeze["sha256"].items():
        path = PROJECT_ROOT / relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"{relative_path}: expected {expected}, found {actual}")
    if mismatches:
        raise ValueError("Frozen test input mismatch:\n" + "\n".join(mismatches))


def load_designs() -> dict[str, dict[str, Any]]:
    payload = json.loads((PROJECT_ROOT / "configs" / "designs.json").read_text())
    if payload["status"] != "frozen_after_gate8_before_test":
        raise ValueError("Designs are not frozen for test evaluation")
    return payload["designs"]


def simulate_design(
    design: dict[str, Any], object_image: np.ndarray
) -> np.ndarray:
    if design["kind"] == "zernike":
        return deterministic_sensor(
            np.asarray(design["phase_parameters"], dtype=np.float32),
            object_image,
            DEPTHS_UM,
        )
    if design["kind"] == "cubic":
        sensor = simulate_cubic_sensor(
            jnp.asarray(design["rms_strength_radians"], dtype=jnp.float32),
            jnp.asarray(object_image[None], dtype=jnp.float32),
            jnp.asarray(DEPTHS_UM),
        )
        return np.maximum(np.asarray(sensor.block_until_ready()[0]), 0.0)
    raise ValueError(f"Unknown optical design kind: {design['kind']}")


def metric_row(
    *,
    source_image_id: str,
    design_name: str,
    condition: dict[str, Any],
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    valid_target, valid_prediction = valid_region_labels(target, prediction, margin=48)
    metrics = instance_metrics(valid_target, valid_prediction)
    return {
        "source_image_id": source_image_id,
        "design": design_name,
        **condition,
        **metrics.to_dict(),
    }


def evaluate_source_design(
    *,
    source_image_id: str,
    design_name: str,
    deterministic: np.ndarray,
    target: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    transform = load_frozen_transform()
    observer_shape = observer_size(target.shape)
    deterministic_labels = segment_sensor_batch(
        model,
        deterministic,
        observer_shape,
        device=device,
        maximum_batch_size=4,
        transform=transform,
    )
    deterministic_labels = resample_labels(deterministic_labels, target.shape)
    rows = [
        metric_row(
            source_image_id=source_image_id,
            design_name=design_name,
            condition={
                "kind": "deterministic",
                "depth_um": float(depth),
                "expected_photons": 100,
                "replicate": -1,
            },
            target=target,
            prediction=prediction,
        )
        for depth, prediction in zip(DEPTHS_UM, deterministic_labels, strict=True)
    ]

    poisson, conditions = poisson_sensor_batch(
        deterministic,
        source_image_id=source_image_id,
        depths_um=DEPTHS_UM,
    )
    poisson_labels = segment_sensor_batch(
        model,
        poisson,
        observer_shape,
        device=device,
        maximum_batch_size=4,
        transform=transform,
    )
    poisson_labels = resample_labels(poisson_labels, target.shape)
    rows.extend(
        metric_row(
            source_image_id=source_image_id,
            design_name=design_name,
            condition=condition,
            target=target,
            prediction=prediction,
        )
        for condition, prediction in zip(conditions, poisson_labels, strict=True)
    )
    return rows


def write_raw_rows(rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "raw-metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (OUTPUT_DIR / "raw-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select(
    rows: list[dict[str, Any]],
    *,
    design: str,
    kind: str,
    primary_only: bool = False,
    photon_level: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["design"] == design
        and row["kind"] == kind
        and (not primary_only or row["depth_um"] in PRIMARY_DEPTHS_UM)
        and (photon_level is None or row["expected_photons"] == photon_level)
    ]
    if not selected:
        raise ValueError(f"No rows for {design}/{kind}/{photon_level}")
    return selected


def summarize_design(rows: list[dict[str, Any]], design: str) -> dict[str, Any]:
    primary = _select(rows, design=design, kind="deterministic", primary_only=True)
    focus = [
        row
        for row in _select(rows, design=design, kind="deterministic")
        if row["depth_um"] == 0.0
    ]
    by_depth = {
        str(depth): float(
            np.mean(
                [row["panoptic_quality"] for row in primary if row["depth_um"] == depth]
            )
        )
        for depth in sorted(PRIMARY_DEPTHS_UM)
    }
    photon = {
        str(level): float(
            np.mean(
                [
                    row["panoptic_quality"]
                    for row in _select(
                        rows,
                        design=design,
                        kind="poisson",
                        photon_level=level,
                    )
                ]
            )
        )
        for level in (50, 200)
    }
    return {
        "mean_off_focus_pq": float(np.mean([row["panoptic_quality"] for row in primary])),
        "worst_depth_pq": min(by_depth.values()),
        "focus_pq": float(np.mean([row["panoptic_quality"] for row in focus])),
        "mean_off_focus_f1_iou_0_5": float(
            np.mean([row["recognition_quality"] for row in primary])
        ),
        "mean_off_focus_absolute_count_error": float(
            np.mean([row["absolute_count_error"] for row in primary])
        ),
        "mean_off_focus_percentage_count_error": float(
            np.mean([row["percentage_count_error"] for row in primary])
        ),
        "mean_off_focus_foreground_dice": float(
            np.mean([row["foreground_dice"] for row in primary])
        ),
        "off_focus_pq_by_depth": by_depth,
        "poisson_endpoint_mean_pq_by_photon_level": photon,
    }


def paired_pq_comparison(
    rows: list[dict[str, Any]],
    candidate: str,
    reference: str,
    *,
    kind: str = "deterministic",
    photon_level: int | None = None,
) -> dict[str, float | int]:
    primary_only = kind == "deterministic"
    candidate_rows = _select(
        rows,
        design=candidate,
        kind=kind,
        primary_only=primary_only,
        photon_level=photon_level,
    )
    reference_rows = _select(
        rows,
        design=reference,
        kind=kind,
        primary_only=primary_only,
        photon_level=photon_level,
    )

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["source_image_id"],
            row["depth_um"],
            row["expected_photons"],
            row["replicate"],
        )

    candidate_by_key = {key(row): row for row in candidate_rows}
    reference_by_key = {key(row): row for row in reference_rows}
    if candidate_by_key.keys() != reference_by_key.keys():
        raise ValueError(f"Unpaired rows for {candidate} and {reference}")
    keys = sorted(candidate_by_key)
    return grouped_bootstrap_difference(
        np.asarray([item[0] for item in keys]),
        np.asarray([candidate_by_key[item]["panoptic_quality"] for item in keys]),
        np.asarray([reference_by_key[item]["panoptic_quality"] for item in keys]),
        replicates=1000,
        seed=20260901,
    )


def build_report(rows: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    designs = ("clear", "cubic", "image_fidelity", "exact_task", "surrogate")
    summaries = {design: summarize_design(rows, design) for design in designs}
    comparisons = {
        reference: paired_pq_comparison(rows, "exact_task", reference)
        for reference in ("clear", "cubic", "image_fidelity", "surrogate")
    }
    photon_comparisons = {
        str(level): paired_pq_comparison(
            rows,
            "exact_task",
            "clear",
            kind="poisson",
            photon_level=level,
        )
        for level in (50, 200)
    }
    exact = summaries["exact_task"]
    clear = summaries["clear"]
    count_reduction = (
        (
            clear["mean_off_focus_absolute_count_error"]
            - exact["mean_off_focus_absolute_count_error"]
        )
        / clear["mean_off_focus_absolute_count_error"]
        if clear["mean_off_focus_absolute_count_error"] > 0
        else 0.0
    )
    checks = {
        "off_focus_pq_over_clear_at_least_0_05": comparisons["clear"][
            "mean_difference"
        ]
        >= 0.05,
        "off_focus_pq_over_cubic_at_least_0_02": comparisons["cubic"][
            "mean_difference"
        ]
        >= 0.02,
        "off_focus_pq_over_image_at_least_0_02": comparisons["image_fidelity"][
            "mean_difference"
        ]
        >= 0.02,
        "primary_reference_ci_lower_bounds_positive": all(
            comparisons[reference]["ci_lower_95"] > 0
            for reference in ("clear", "cubic", "image_fidelity")
        ),
        "worst_depth_pq_over_clear_at_least_0_04": (
            exact["worst_depth_pq"] - clear["worst_depth_pq"]
        )
        >= 0.04,
        "focus_pq_drop_no_worse_than_0_03": (
            exact["focus_pq"] - clear["focus_pq"]
        )
        >= -0.03,
        "count_error_reduction_at_least_10_percent": count_reduction >= 0.10,
        "exact_over_surrogate_at_least_0_02": comparisons["surrogate"][
            "mean_difference"
        ]
        >= 0.02,
        "exact_over_surrogate_ci_lower_positive": comparisons["surrogate"][
            "ci_lower_95"
        ]
        > 0,
        "positive_at_both_photon_levels": all(
            photon_comparisons[str(level)]["mean_difference"] > 0
            for level in (50, 200)
        ),
    }
    positive_claim = all(checks.values())
    return {
        "gate": "gate9_locked_test_evaluation",
        "source_count": len({row["source_image_id"] for row in rows}),
        "row_count": len(rows),
        "wall_seconds": wall_seconds,
        "summaries": summaries,
        "paired_pq_comparisons": comparisons,
        "poisson_exact_vs_clear": photon_comparisons,
        "off_focus_count_error_reduction_fraction_vs_clear": count_reduction,
        "claim_checks": checks,
        "positive_headline_claim": positive_claim,
        "conclusion": (
            "all frozen positive-claim thresholds passed"
            if positive_claim
            else "one or more frozen positive-claim thresholds failed; report a negative result"
        ),
        "evaluation_completed": True,
    }


def main() -> None:
    verify_freeze()
    started = time.perf_counter()
    designs = load_designs()
    records = records_for_split("test")
    normalization = ObjectNormalization(125.0, 1642.0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    rows: list[dict[str, Any]] = []
    computed_designs = ("clear", "image_fidelity", "exact_task", "surrogate")
    for source_index, record in enumerate(records):
        object_image, target = full_source_inputs(record, normalization)
        for design_name in computed_designs:
            pair_started = time.perf_counter()
            deterministic = simulate_design(designs[design_name], object_image)
            pair_rows = evaluate_source_design(
                source_image_id=record.image_id,
                design_name=design_name,
                deterministic=deterministic,
                target=target,
                model=model,
                device=device,
            )
            rows.extend(pair_rows)
            if design_name == "clear":
                rows.extend([{**row, "design": "cubic"} for row in pair_rows])
            write_raw_rows(rows)
            print(
                json.dumps(
                    {
                        "source": source_index + 1,
                        "source_count": len(records),
                        "source_image_id": record.image_id,
                        "design": design_name,
                        "pair_seconds": time.perf_counter() - pair_started,
                        "rows_written": len(rows),
                    }
                ),
                flush=True,
            )
    report = build_report(rows, time.perf_counter() - started)
    (OUTPUT_DIR / "test-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
