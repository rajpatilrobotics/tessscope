"""Evaluate clear-microscope spectral focus on train/validation BBBC006 stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tessscope.v2.autofocus.features import SpectralFeatureConfig, spectral_features
from tessscope.v2.autofocus.metrics import focus_metrics
from tessscope.v2.autofocus.ridge import ridge_predict
from tessscope.v2.data.bbbc006 import DEPTHS_UM, records_for_split
from tessscope.v2.data.preprocess import (
    accepted_registration_fields,
    load_global_normalization,
    load_registration_table,
    registered_normalized_crop,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "artifacts" / "runs" / "v2" / "data" / "clear-features.npz"
OUTPUT = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "gate2" / "clear-autofocus.json"
)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def build_cache(crop_size: int) -> dict[str, np.ndarray]:
    normalization = load_global_normalization()
    registrations = load_registration_table()
    accepted_fields = accepted_registration_fields()
    feature_config = SpectralFeatureConfig(radial_bins=10, angular_bins=12)
    features = []
    depths = []
    splits = []
    wells = []
    fields = []
    for split in ("training", "validation"):
        records = records_for_split(split)
        for completed, record in enumerate(records, start=1):
            if record.field_id not in accepted_fields:
                continue
            stack = registered_normalized_crop(
                record,
                normalization,
                registrations,
                size=crop_size,
            )
            stack_features, _ = spectral_features(stack, feature_config)
            features.append(stack_features)
            depths.extend(DEPTHS_UM)
            splits.extend([split] * len(DEPTHS_UM))
            wells.extend([record.well] * len(DEPTHS_UM))
            fields.extend([record.field_id] * len(DEPTHS_UM))
            if completed % 25 == 0 or completed == len(records):
                print(
                    json.dumps({"feature_split": split, "completed": completed}),
                    flush=True,
                )
    payload = {
        "features": np.concatenate(features).astype(np.float32),
        "depth_um": np.asarray(depths, dtype=np.float32),
        "split": np.asarray(splits),
        "well": np.asarray(wells),
        "field": np.asarray(fields),
        "crop_size": np.asarray(crop_size),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    return payload


def main() -> None:
    args = parse_args()
    if args.reuse_cache:
        with np.load(CACHE) as cached:
            payload = {key: cached[key] for key in cached.files}
    else:
        payload = build_cache(args.crop_size)
    if int(payload["crop_size"]) != args.crop_size:
        raise ValueError("Cached autofocus crop size differs from requested crop size")
    training = payload["split"] == "training"
    validation = payload["split"] == "validation"
    minimum_training = int(0.90 * 288 * 2 * 7)
    minimum_validation = int(0.90 * 48 * 2 * 7)
    if int(np.count_nonzero(training)) < minimum_training:
        raise ValueError("Clear autofocus cache lost more than 10% of training fields")
    if int(np.count_nonzero(validation)) < minimum_validation:
        raise ValueError("Clear autofocus cache lost more than 10% of validation fields")
    candidates = []
    for ridge_lambda in RIDGE_GRID:
        result = ridge_predict(
            payload["features"][training],
            payload["depth_um"][training],
            payload["features"][validation],
            ridge_lambda,
        )
        metrics = focus_metrics(
            payload["depth_um"][validation], result.predictions
        )
        candidates.append({"ridge_lambda": ridge_lambda, **metrics})
    selected = min(candidates, key=lambda row: (row["mae_um"], row["ridge_lambda"]))
    report = {
        "dataset": "BBBC006v1",
        "design": "clear_real_microscope",
        "splits": {"fit": "training wells", "selection": "validation wells"},
        "test_accessed": False,
        "crop_size": args.crop_size,
        "feature_grid": {"radial_bins": 10, "angular_bins": 12},
        "ridge_candidates": candidates,
        "selected": selected,
        "interpretation": (
            "Clear circular-pupil defocus is expected to be approximately sign-symmetric; "
            "this is a baseline feasibility diagnostic, not the joint-pupil focus gate."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
