"""Run the frozen training-only fixed-total-photon exposure audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
import yaml

from tessscope.evaluation.metrics import instance_metrics, valid_region_labels
from tessscope.evaluation.route import segment_sensor_batch
from tessscope.observer.instanseg import load_frozen_model
from tessscope.observer.loss import ObserverTransform
from tessscope.v2.autofocus.features import spectral_features
from tessscope.v2.autofocus.ridge import ridge_predict
from tessscope.v2.optimization.served import DEPTHS, V2SystemCalibration
from tessscope.v2_1.optics.model import simulate_noisy_sensor_b7
from tessscope.v2_6.diagnostics import paired_well_evidence
from tessscope.v2_6.exposure import (
    collect_exposure_patches,
    exposure_photons,
    select_development_fraction,
    summarize_exposure_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT_ROOT / "configs" / "v2_6" / "exposure-audit.yaml"
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_6"
    / "diagnostics"
    / "exposure-audit.json"
)


def simulate(
    parameters: np.ndarray,
    objects: np.ndarray,
    depths: np.ndarray,
    noise: np.ndarray,
    expected_photons: float,
    calibration: V2SystemCalibration,
) -> np.ndarray:
    return np.asarray(
        simulate_noisy_sensor_b7(
            jnp.asarray(parameters, dtype=jnp.float32),
            jnp.asarray(objects, dtype=jnp.float32),
            jnp.asarray(depths, dtype=jnp.float32),
            jnp.asarray(noise, dtype=jnp.float32),
            expected_photons=expected_photons,
            exposure_gain=calibration.exposure_gain,
            depth_scale=calibration.depth_scale,
            axial_offset_um=calibration.axial_offset_um,
        )
    )


def evaluate_policy(
    *,
    design: dict,
    partition: str,
    patches,
    fraction: float,
    noise_seed: int,
    calibration: V2SystemCalibration,
    model,
    device: torch.device,
    transform: ObserverTransform,
) -> list[dict]:
    first_photons, second_photons = exposure_photons(400.0, fraction)
    parameters = np.asarray(design["parameters"], dtype=np.float32)
    result = []
    for query_index, query in enumerate(patches):
        support = [
            patches[(query_index + 1) % len(patches)],
            patches[(query_index + 2) % len(patches)],
        ]
        objects = np.stack(
            [support[0].object_image, support[1].object_image, query.object_image]
        ).astype(np.float32)
        generator = np.random.default_rng(noise_seed + query_index * 1000)
        first_noise = generator.normal(size=(3, 7, 256, 256)).astype(np.float32)
        second_noise = generator.normal(size=(1, 7, 256, 256)).astype(np.float32)
        first = simulate(
            parameters,
            objects,
            DEPTHS,
            first_noise,
            first_photons,
            calibration,
        )
        features, _ = spectral_features(first.reshape(-1, 256, 256))
        ridge = ridge_predict(
            features[:14],
            np.tile(DEPTHS, 2),
            features[14:],
            ridge_lambda=calibration.ridge_lambda,
        )
        residual_depths = DEPTHS + np.clip(-ridge.predictions, -6.0, 6.0)
        second = simulate(
            parameters,
            query.object_image[None].astype(np.float32),
            residual_depths.astype(np.float32),
            second_noise,
            second_photons,
            calibration,
        )[0]
        first_predictions = segment_sensor_batch(
            model,
            first[2],
            query.instance_labels.shape,
            device=device,
            maximum_batch_size=4,
            transform=transform,
        )
        second_predictions = segment_sensor_batch(
            model,
            second,
            query.instance_labels.shape,
            device=device,
            maximum_batch_size=4,
            transform=transform,
        )
        first_mean = float(np.mean(first) * first_photons)
        second_mean = float(np.mean(second) * second_photons)
        for depth, predicted, residual, before, after in zip(
            DEPTHS,
            ridge.predictions,
            residual_depths,
            first_predictions,
            second_predictions,
            strict=True,
        ):
            if depth == 0.0:
                continue
            target, before_valid = valid_region_labels(
                query.instance_labels, before, margin=62
            )
            _, after_valid = valid_region_labels(query.instance_labels, after, margin=62)
            result.append(
                {
                    "design": design["name"],
                    "partition": partition,
                    "well": query.well,
                    "field_id": query.field_id,
                    "noise_seed": noise_seed,
                    "first_fraction": fraction,
                    "first_expected_photons": first_photons,
                    "second_expected_photons": second_photons,
                    "original_depth_um": float(depth),
                    "predicted_depth_um": float(predicted),
                    "residual_depth_um": float(residual),
                    "before_pq": instance_metrics(target, before_valid).panoptic_quality,
                    "after_pq": instance_metrics(target, after_valid).panoptic_quality,
                    "first_photon_mean": first_mean,
                    "second_photon_mean": second_mean,
                }
            )
    return result


def average_noise_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["well"], row["field_id"], row["original_depth_um"])
        grouped.setdefault(key, []).append(row)
    averaged = []
    for key, values in sorted(grouped.items()):
        template = values[0]
        averaged.append(
            {
                "well": key[0],
                "field_id": key[1],
                "original_depth_um": key[2],
                "before_pq": float(np.mean([row["before_pq"] for row in values])),
                "after_pq": float(np.mean([row["after_pq"] for row in values])),
                "residual_depth_um": float(
                    np.mean([row["residual_depth_um"] for row in values])
                ),
                "design": template["design"],
            }
        )
    return averaged


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"V2.6 exposure audit already exists: {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text())
    partition_path = (
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "v2_6"
        / "training-development-confirmation.json"
    )
    if hashlib.sha256(partition_path.read_bytes()).hexdigest() != contract["data"][
        "source_partition_sha256"
    ]:
        raise ValueError("Frozen v2.6 partition hash mismatch")
    designs = list(contract["systems"].values())
    fractions = contract["photon_policy"]["first_exposure_fraction_grid"]
    seeds = contract["photon_policy"]["fixed_standard_normal_seeds"]
    calibration = V2SystemCalibration.load()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_frozen_model(device)
    transform = ObserverTransform(
        mode="affine", offset=0.0, scale=calibration.exposure_gain
    )
    all_rows = []
    patch_registry = {}
    total = 2 * len(designs) * len(fractions) * len(seeds)
    index = 0
    for partition in ("development", "confirmation"):
        patches = collect_exposure_patches(partition)
        patch_registry[partition] = [
            {
                "well": patch.well,
                "field_id": patch.field_id,
                "origin_yx": list(patch.origin_yx),
            }
            for patch in patches
        ]
        for design in designs:
            for fraction in fractions:
                for seed in seeds:
                    index += 1
                    rows = evaluate_policy(
                        design=design,
                        partition=partition,
                        patches=patches,
                        fraction=fraction,
                        noise_seed=seed,
                        calibration=calibration,
                        model=model,
                        device=device,
                        transform=transform,
                    )
                    all_rows.extend(rows)
                    print(
                        json.dumps(
                            {
                                "index": index,
                                "total": total,
                                "partition": partition,
                                "design": design["name"],
                                "fraction": fraction,
                                "seed": seed,
                                "corrected_pq": round(
                                    summarize_exposure_rows(rows)["corrected_pq"], 6
                                ),
                            }
                        ),
                        flush=True,
                    )
    by_design_partition = {}
    for design in designs:
        name = design["name"]
        development = [
            row
            for row in all_rows
            if row["design"] == name and row["partition"] == "development"
        ]
        selection = select_development_fraction(development)
        fraction = selection["selected_first_fraction"]
        confirmation = [
            row
            for row in all_rows
            if row["design"] == name
            and row["partition"] == "confirmation"
            and row["first_fraction"] == fraction
        ]
        by_design_partition[name] = {
            "development_selection": selection,
            "confirmation_summary": summarize_exposure_rows(confirmation),
            "confirmation_rows_averaged_over_noise": average_noise_rows(confirmation),
        }
    candidate_name = contract["systems"]["candidate"]["name"]
    baseline_name = contract["systems"]["baseline"]["name"]
    candidate = by_design_partition[candidate_name]
    baseline = by_design_partition[baseline_name]
    evidence = paired_well_evidence(
        candidate["confirmation_rows_averaged_over_noise"],
        baseline["confirmation_rows_averaged_over_noise"],
    )
    candidate_summary = candidate["confirmation_summary"]
    checks = {
        "gain_at_least_0_005": evidence["mean_difference"] >= 0.005,
        "bootstrap_lower_positive": evidence["ci_lower_95"] > 0.0,
        "positive_well_fraction_at_least_0_60": evidence["positive_well_fraction"]
        >= 0.60,
        "leave_one_out_always_positive": evidence[
            "minimum_leave_one_well_out_mean_difference"
        ]
        > 0.0,
        "candidate_focus_mae_at_most_1_0": candidate_summary["focus_mae_um"] <= 1.0,
        "candidate_first_pq_preserved": candidate_summary["first_pq"]
        >= candidate["development_selection"]["half_split_first_pq"] - 0.005,
        "prior_oracle_gate_passed": False,
    }
    report = {
        "status": "complete_training_development_confirmation_exposure_audit",
        "test_accessed": False,
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "patch_registry": patch_registry,
        "systems": by_design_partition,
        "candidate_vs_baseline_confirmation": evidence,
        "route_checks": checks,
        "controller_exposure_route_supported": all(checks.values()),
        "rows": all_rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "candidate_fraction": candidate["development_selection"][
                    "selected_first_fraction"
                ],
                "baseline_fraction": baseline["development_selection"][
                    "selected_first_fraction"
                ],
                "candidate_confirmation": candidate_summary,
                "baseline_confirmation": baseline["confirmation_summary"],
                "gain": evidence["mean_difference"],
                "supported": all(checks.values()),
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
