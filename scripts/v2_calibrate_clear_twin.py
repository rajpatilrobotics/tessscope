"""Fit a bounded clear-PSF depth scale to train OTF ratios and gate on validation."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tessscope.optics.model import psf_sensor
from tessscope.v2.calibration.transfer import (
    fit_transfer_amplitude,
    pearson_correlation,
    radial_otf_log_power,
    transfer_ratios,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2" / "data" / "clear-features.npz"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2"
    / "gate3"
    / "clear-calibration-offset.json"
)
DEPTHS = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
DEPTH_SCALE_GRID = np.linspace(0.75, 1.25, 21)
AXIAL_OFFSET_GRID = np.linspace(-6.0, 6.0, 25)
SELECTED_RADIAL_BINS = slice(1, 7)


@jax.jit
def clear_psfs(depths_um: jax.Array) -> jax.Array:
    parameters = jnp.zeros(6, dtype=jnp.float32)
    return jax.vmap(lambda depth: psf_sensor(parameters, depth))(depths_um)


def observed_transfer_curves(
    features: np.ndarray, split_values: np.ndarray, split: str
) -> np.ndarray:
    selected = np.asarray(features[split_values == split], dtype=np.float64)
    if len(selected) % 7 != 0 or selected.shape[1] != 120:
        raise ValueError("Expected seven 10×12 feature rows per field")
    by_field = selected.reshape(-1, 7, 10, 12).mean(axis=-1)
    ratios = by_field - by_field[:, 3:4]
    return np.median(ratios, axis=0)


def simulated_curves(depth_scale: float, axial_offset_um: float) -> np.ndarray:
    physical_depths = DEPTHS * depth_scale + axial_offset_um
    psfs = np.asarray(clear_psfs(jnp.asarray(physical_depths)))
    return transfer_ratios(radial_otf_log_power(psfs, radial_bins=10))


def main() -> None:
    with np.load(FEATURE_CACHE) as cached:
        features = cached["features"]
        split_values = cached["split"]
    training = observed_transfer_curves(features, split_values, "training")
    validation = observed_transfer_curves(features, split_values, "validation")
    selected_depths = np.r_[0:3, 4:7]
    candidates = []
    simulated_by_parameters = {}
    for depth_scale in DEPTH_SCALE_GRID:
        for axial_offset_um in AXIAL_OFFSET_GRID:
            key = (float(depth_scale), float(axial_offset_um))
            simulated = simulated_curves(*key)
            simulated_by_parameters[key] = simulated
            fit = fit_transfer_amplitude(
                training[selected_depths, SELECTED_RADIAL_BINS],
                simulated[selected_depths, SELECTED_RADIAL_BINS],
            )
            candidates.append(
                {
                    "depth_scale": key[0],
                    "axial_offset_um": key[1],
                    "amplitude": fit.amplitude,
                    "training_correlation": fit.correlation,
                    "training_rmse": fit.root_mean_squared_error,
                }
            )
    selected = min(
        candidates,
        key=lambda row: (-row["training_correlation"], row["training_rmse"]),
    )
    simulated = simulated_by_parameters[
        (selected["depth_scale"], selected["axial_offset_um"])
    ]
    fitted = selected["amplitude"] * simulated
    validation_transfer_correlation = pearson_correlation(
        validation[selected_depths, SELECTED_RADIAL_BINS],
        fitted[selected_depths, SELECTED_RADIAL_BINS],
    )
    validation_depth_curve = validation[:, SELECTED_RADIAL_BINS].mean(axis=1)
    simulated_depth_curve = fitted[:, SELECTED_RADIAL_BINS].mean(axis=1)
    validation_depth_curve_correlation = pearson_correlation(
        validation_depth_curve, simulated_depth_curve
    )
    passed = validation_depth_curve_correlation >= 0.90
    report = {
        "dataset": "BBBC006v1",
        "fit_split": "training wells only",
        "gate_split": "validation wells only",
        "test_accessed": False,
        "model": {
            "numerical_aperture": 0.45,
            "wavelength_um": 0.461,
            "object_sampling_um": 0.645,
            "fitted_depth_scale": selected["depth_scale"],
            "fitted_axial_offset_um": selected["axial_offset_um"],
            "fitted_transfer_amplitude": selected["amplitude"],
            "fixed_aberration_coefficients_radians": [0.0] * 6,
            "background_policy": "global training p0.1 offset in normalization manifest",
            "read_noise_policy": "not identifiable from BBBC006; fixed to zero",
        },
        "selection_radial_bins": [1, 2, 3, 4, 5, 6],
        "candidate_fits": candidates,
        "selected_training_fit": selected,
        "observed_training_transfer": training.tolist(),
        "observed_validation_transfer": validation.tolist(),
        "fitted_simulated_transfer": fitted.tolist(),
        "validation_transfer_correlation": validation_transfer_correlation,
        "validation_depth_curve": validation_depth_curve.tolist(),
        "simulated_depth_curve": simulated_depth_curve.tolist(),
        "validation_depth_curve_correlation": validation_depth_curve_correlation,
        "threshold": 0.90,
        "passed": bool(passed),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selected_depth_scale": selected["depth_scale"],
                "selected_axial_offset_um": selected["axial_offset_um"],
                "validation_depth_curve_correlation": validation_depth_curve_correlation,
                "passed": passed,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit("Held-out clear digital-twin calibration gate failed")


if __name__ == "__main__":
    main()
