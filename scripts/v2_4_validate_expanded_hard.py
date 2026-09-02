"""Run frozen expanded hard validation for exact and stopped v2.4 pupils."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from tessscope.evaluation.metrics import instance_metrics, valid_region_labels
from tessscope.evaluation.route import segment_sensor_batch
from tessscope.observer.instanseg import load_frozen_model
from tessscope.observer.loss import ObserverTransform
from tessscope.v2.autofocus.features import spectral_features
from tessscope.v2.autofocus.metrics import focus_metrics
from tessscope.v2.autofocus.ridge import ridge_predict
from tessscope.v2.optimization.served import (
    DEPTHS,
    V2SystemCalibration,
    collect_patches,
)
from tessscope.v2_1.evaluation import paired_hard_pq_bootstrap
from tessscope.v2_1.optics.model import simulate_noisy_sensor_b7
from tessscope.v2_4.hard import paired_corrected_pq_bootstrap, well_stability

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION = PROJECT_ROOT / "configs" / "v2_4" / "selected-checkpoints.json"
STOPPED = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "stopped-stage-matched.json"
)
V2_1_HARD = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "validation"
    / "expanded-hard-validation.json"
)
V2_2_HARD = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "validation"
    / "piecewise-hard-frontier.json"
)
DERIVATIVE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_3"
    / "gates"
    / "closed-loop-derivative.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)
CHECKPOINT_ROOT = PROJECT_ROOT / "artifacts" / "runtime-runs" / "v2_4-hard"
PIECEWISE_NAME = "v2_2-piecewise-028"
CLEAR_NAME = "clear"
SEGMENTATION_NAME = "b7_segmentation_only"


def simulate(
    parameters: np.ndarray,
    objects: np.ndarray,
    depths: np.ndarray,
    calibration: V2SystemCalibration,
) -> np.ndarray:
    noise = np.zeros((len(objects), len(depths), 256, 256), dtype=np.float32)
    return np.maximum(
        np.asarray(
            simulate_noisy_sensor_b7(
                jnp.asarray(parameters, dtype=jnp.float32),
                jnp.asarray(objects, dtype=jnp.float32),
                jnp.asarray(depths, dtype=jnp.float32),
                jnp.asarray(noise),
                expected_photons=calibration.expected_photons,
                exposure_gain=calibration.exposure_gain,
                depth_scale=calibration.depth_scale,
                axial_offset_um=calibration.axial_offset_um,
            )
        ),
        0.0,
    )


def summarize(rows: list[dict], photon_means: list[float]) -> dict:
    off_focus = [row for row in rows if row["depth_um"] != 0.0]
    hard = [row for row in off_focus if row["hard_dense_patch"]]
    targets = np.asarray([row["depth_um"] for row in rows])
    predictions = np.asarray([row["predicted_depth_um"] for row in rows])
    summary = {
        "mean_photon_count": float(np.mean(photon_means)),
        "sample_count": len(rows),
        "hard_off_focus_sample_count": len(hard),
        **focus_metrics(targets, predictions),
    }
    for metric, label in (
        ("panoptic_quality", "pq"),
        ("recognition_quality", "rq"),
        ("segmentation_quality", "sq"),
        ("foreground_dice", "dice"),
        ("absolute_count_error", "absolute_count_error"),
        ("percentage_count_error", "percentage_count_error"),
    ):
        summary[f"hard_dense_off_focus_{label}"] = float(
            np.mean([row[metric] for row in hard])
        )
    return summary


def evaluate_design(
    design: dict,
    patches,
    hard_fields: set[str],
    calibration: V2SystemCalibration,
    model,
    device: torch.device,
    transform: ObserverTransform,
) -> dict:
    parameters = np.asarray(design["parameters"], dtype=np.float32)
    rows = []
    corrected_rows = []
    photon_means = []
    for query_index, query in enumerate(patches):
        support = [
            patches[(query_index + 1) % len(patches)],
            patches[(query_index + 2) % len(patches)],
        ]
        objects = np.stack(
            [support[0].object_image, support[1].object_image, query.object_image]
        ).astype(np.float32)
        sensor = simulate(parameters, objects, DEPTHS, calibration)
        photon_means.append(float(np.mean(sensor) * calibration.expected_photons))
        features, _ = spectral_features(sensor.reshape(-1, 256, 256))
        ridge = ridge_predict(
            features[:14],
            np.tile(DEPTHS, 2),
            features[14:],
            ridge_lambda=calibration.ridge_lambda,
        )
        predictions = segment_sensor_batch(
            model,
            sensor[2],
            query.instance_labels.shape,
            device=device,
            maximum_batch_size=4,
            transform=transform,
        )
        query_rows = []
        for depth, predicted_depth, prediction in zip(
            DEPTHS, ridge.predictions, predictions, strict=True
        ):
            target, valid_prediction = valid_region_labels(
                query.instance_labels, prediction, margin=62
            )
            row = {
                "design": design["name"],
                "well": query.well,
                "field_id": query.field_id,
                "depth_um": float(depth),
                "predicted_depth_um": float(predicted_depth),
                "stage_action_um": float(np.clip(-predicted_depth, -6.0, 6.0)),
                "hard_dense_patch": query.field_id in hard_fields,
                "valid_instance_count": int(np.count_nonzero(query.valid_objects)),
                **instance_metrics(target, valid_prediction).to_dict(),
            }
            rows.append(row)
            query_rows.append(row)
        residual_depths = DEPTHS + np.clip(-ridge.predictions, -6.0, 6.0)
        if query.field_id not in hard_fields:
            continue
        corrected_sensor = simulate(
            parameters,
            query.object_image[None].astype(np.float32),
            residual_depths.astype(np.float32),
            calibration,
        )[0]
        corrected_predictions = segment_sensor_batch(
            model,
            corrected_sensor,
            query.instance_labels.shape,
            device=device,
            maximum_batch_size=4,
            transform=transform,
        )
        for original, residual, prediction in zip(
            query_rows, residual_depths, corrected_predictions, strict=True
        ):
            if original["depth_um"] == 0.0:
                continue
            target, valid_prediction = valid_region_labels(
                query.instance_labels, prediction, margin=62
            )
            corrected_rows.append(
                {
                    "design": design["name"],
                    "well": query.well,
                    "field_id": query.field_id,
                    "original_depth_um": original["depth_um"],
                    "residual_depth_um": float(residual),
                    "hard_dense_patch": True,
                    "before_pq": original["panoptic_quality"],
                    "after_pq": instance_metrics(
                        target, valid_prediction
                    ).panoptic_quality,
                }
            )
    summary = summarize(rows, photon_means)
    hard_primary = [
        row for row in rows if row["hard_dense_patch"] and row["depth_um"] != 0.0
    ]
    stage = {
        "mean_absolute_depth_before_um": float(
            np.mean([abs(row["depth_um"]) for row in rows])
        ),
        "mean_absolute_depth_after_um": float(
            np.mean(
                [abs(row["depth_um"] + row["stage_action_um"]) for row in rows]
            )
        ),
        "hard_off_focus_pq_before": float(
            np.mean([row["panoptic_quality"] for row in hard_primary])
        ),
        "hard_off_focus_pq_after": float(
            np.mean([row["after_pq"] for row in corrected_rows])
        ),
        "fraction_hard_frames_improved": float(
            np.mean([row["after_pq"] > row["before_pq"] for row in corrected_rows])
        ),
    }
    return {
        "name": design["name"],
        "role": design["role"],
        "summary": summary,
        "stage_correction": stage,
        "rows": rows,
        "corrected_rows": corrected_rows,
    }


def load_or_evaluate(
    design,
    patches,
    hard_fields,
    calibration,
    model,
    device,
    transform,
    checkpoint_root,
):
    checkpoint = checkpoint_root / f"{design['name']}.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())
    result = evaluate_design(
        design, patches, hard_fields, calibration, model, device, transform
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(result) + "\n")
    return result


def design_registry(selection: dict, stopped: dict) -> tuple[list[dict], dict[str, str]]:
    designs = [
        {
            "name": row["name"],
            "parameters": row["parameters"],
            "role": "frozen_v2_4_exact_checkpoint",
        }
        for row in selection["selected"]
    ]
    matched_stopped = {}
    for pair in stopped["pairs"]:
        stopped_row = pair["stopped"]
        designs.append(
            {
                "name": stopped_row["name"],
                "parameters": stopped_row["final_parameters"],
                "role": "matched_v2_4_stopped_stage",
            }
        )
        matched_stopped[pair["exact"]["name"]] = stopped_row["name"]
    return designs, matched_stopped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="write the report to an isolated path",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=CHECKPOINT_ROOT,
        help="write resumable per-design evaluations to an isolated directory",
    )
    return parser.parse_args()


def project_output(path: Path) -> Path:
    output = path if path.is_absolute() else PROJECT_ROOT / path
    output = output.resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Reproduction output must stay inside the project directory")
    return output


def main() -> None:
    args = parse_args()
    output = project_output(args.output)
    checkpoint_root = project_output(args.checkpoint_root)
    if output.exists():
        raise SystemExit(f"V2.4 hard artifact already exists: {output}")
    selection = json.loads(SELECTION.read_text())
    stopped = json.loads(STOPPED.read_text())
    old = json.loads(V2_1_HARD.read_text())
    piecewise = json.loads(V2_2_HARD.read_text())
    derivative = json.loads(DERIVATIVE.read_text())
    designs, matched_stopped = design_registry(selection, stopped)
    patches = collect_patches("validation", 45, seed=53)
    expected_fields = [
        row["field_id"]
        for row in old["rows"]
        if row["design"] == CLEAR_NAME and row["depth_um"] == 0.0
    ]
    if [patch.field_id for patch in patches] != expected_fields:
        raise SystemExit("V2.4 patch order differs from frozen expanded protocol")
    hard_fields = set(old["hard_definition"]["hard_field_ids"])
    if len(hard_fields) != 27:
        raise SystemExit("V2.4 hard-density subset must contain 27 fields")
    calibration = V2SystemCalibration.load()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = ObserverTransform(
        mode="affine", offset=0.0, scale=calibration.exposure_gain
    )
    results = []
    for index, design in enumerate(designs, start=1):
        result = load_or_evaluate(
            design,
            patches,
            hard_fields,
            calibration,
            model,
            device,
            transform,
            checkpoint_root,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(designs),
                    "design": design["name"],
                    "role": design["role"],
                    "hard_pq": round(
                        result["summary"]["hard_dense_off_focus_pq"], 6
                    ),
                    "mae_um": round(result["summary"]["mae_um"], 6),
                    "corrected_pq": round(
                        result["stage_correction"]["hard_off_focus_pq_after"], 6
                    ),
                }
            ),
            flush=True,
        )
    by_name = {result["name"]: result for result in results}
    new_rows = [row for result in results for row in result["rows"]]
    new_corrected = [row for result in results for row in result["corrected_rows"]]
    baseline_rows = [
        row
        for row in old["rows"]
        if row["design"] in {CLEAR_NAME, SEGMENTATION_NAME}
    ]
    piecewise_corrected = [
        row
        for row in piecewise["corrected_rows"]
        if row["design"] == PIECEWISE_NAME
    ]
    exact_names = [row["name"] for row in selection["selected"]]
    gates = {}
    evidence = {}
    clear_summary = old["summaries"][CLEAR_NAME]
    segmentation_summary = old["summaries"][SEGMENTATION_NAME]
    piecewise_stage = piecewise["stage_correction"][PIECEWISE_NAME]
    for exact_name in exact_names:
        exact = by_name[exact_name]
        stopped_name = matched_stopped[exact_name]
        first_clear = paired_hard_pq_bootstrap(
            [*baseline_rows, *new_rows], exact_name, CLEAR_NAME
        )
        corrected_piecewise = paired_corrected_pq_bootstrap(
            new_corrected,
            piecewise_corrected,
            exact_name,
            PIECEWISE_NAME,
        )
        corrected_stopped = paired_corrected_pq_bootstrap(
            new_corrected,
            new_corrected,
            exact_name,
            stopped_name,
        )
        stability = {
            "first_vs_clear": well_stability(
                new_rows,
                baseline_rows,
                candidate_name=exact_name,
                reference_name=CLEAR_NAME,
                depth_key="depth_um",
                metric_key="panoptic_quality",
            ),
            "corrected_vs_piecewise": well_stability(
                new_corrected,
                piecewise_corrected,
                candidate_name=exact_name,
                reference_name=PIECEWISE_NAME,
                depth_key="original_depth_um",
                metric_key="after_pq",
            ),
            "corrected_vs_stopped": well_stability(
                new_corrected,
                new_corrected,
                candidate_name=exact_name,
                reference_name=stopped_name,
                depth_key="original_depth_um",
                metric_key="after_pq",
            ),
        }
        summary = exact["summary"]
        stage = exact["stage_correction"]
        candidate_gates = {
            "first_hard_pq_gain_over_clear_at_least_0_01": bool(
                summary["hard_dense_off_focus_pq"]
                - clear_summary["hard_dense_off_focus_pq"]
                >= 0.01
            ),
            "first_clear_bootstrap_lower_positive": bool(
                first_clear["ci_lower_95"] > 0.0
            ),
            "first_hard_pq_drop_from_segmentation_at_most_0_01": bool(
                segmentation_summary["hard_dense_off_focus_pq"]
                - summary["hard_dense_off_focus_pq"]
                <= 0.01
            ),
            "signed_direction_at_least_0_90": bool(
                summary["signed_direction_accuracy"] >= 0.90
            ),
            "focus_mae_at_most_2_um": bool(summary["mae_um"] <= 2.0),
            "stage_reduces_absolute_defocus": bool(
                stage["mean_absolute_depth_after_um"]
                < stage["mean_absolute_depth_before_um"]
            ),
            "corrected_hard_pq_exceeds_first": bool(
                stage["hard_off_focus_pq_after"]
                > stage["hard_off_focus_pq_before"]
            ),
            "corrected_gain_over_piecewise_at_least_0_005": bool(
                stage["hard_off_focus_pq_after"]
                - piecewise_stage["hard_off_focus_pq_after"]
                >= 0.005
            ),
            "piecewise_bootstrap_lower_positive": bool(
                corrected_piecewise["ci_lower_95"] > 0.0
            ),
            "corrected_gain_over_stopped_at_least_0_005": bool(
                corrected_stopped["mean_difference"] >= 0.005
            ),
            "stopped_bootstrap_lower_positive": bool(
                corrected_stopped["ci_lower_95"] > 0.0
            ),
            "full_loop_derivative_gate_passed": bool(derivative["passed_v2_3"]),
            "positive_photon_means": bool(
                summary["mean_photon_count"] > 0.0
                and exact["stage_correction"]["hard_off_focus_pq_after"] >= 0.0
            ),
            "well_stability_all_comparisons": bool(
                all(row["passed"] for row in stability.values())
            ),
        }
        candidate_gates["passed_all_hard_gates"] = all(candidate_gates.values())
        gates[exact_name] = candidate_gates
        evidence[exact_name] = {
            "matched_stopped_name": stopped_name,
            "first_vs_clear": first_clear,
            "corrected_vs_piecewise": corrected_piecewise,
            "corrected_vs_stopped": corrected_stopped,
            "well_stability": stability,
        }
    passed_names = [name for name in exact_names if gates[name]["passed_all_hard_gates"]]
    selected_for_derivative_free = (
        max(
            passed_names,
            key=lambda name: (
                by_name[name]["stage_correction"]["hard_off_focus_pq_after"],
                by_name[name]["summary"]["hard_dense_off_focus_pq"],
                -by_name[name]["summary"]["mae_um"],
            ),
        )
        if passed_names
        else None
    )
    report = {
        "status": "complete_expanded_hard_validation_only",
        "test_accessed": False,
        "coverage": old["coverage"],
        "hard_definition": old["hard_definition"],
        "design_registry": designs,
        "summaries": {name: result["summary"] for name, result in by_name.items()},
        "stage_correction": {
            name: result["stage_correction"] for name, result in by_name.items()
        },
        "frozen_references": {
            "clear": clear_summary,
            "segmentation_only": segmentation_summary,
            "piecewise_028": piecewise["summaries"][PIECEWISE_NAME],
            "piecewise_028_stage": piecewise_stage,
        },
        "paired_evidence": evidence,
        "gates": gates,
        "hard_passed_names": passed_names,
        "selected_for_matched_derivative_free": selected_for_derivative_free,
        "next_gate": (
            "matched_derivative_free_control"
            if selected_for_derivative_free
            else "freeze_negative_v2_4_without_test_access"
        ),
        "rows": new_rows,
        "corrected_rows": new_corrected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "hard_passed_names": passed_names,
                "selected_for_matched_derivative_free": selected_for_derivative_free,
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
