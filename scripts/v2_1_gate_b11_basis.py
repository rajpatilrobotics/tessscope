"""Record B11 convention, support, smoothness, and quantization diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tessscope.v2.optimization.served import DEPTHS
from tessscope.v2_1.optics.model import (
    B11_NOLL_INDICES,
    psf_support_energy_fraction_b11,
    unconstrained_from_coefficients_b11,
    zernike_basis_b11,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTED = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-projected-gradient-candidates.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "gates"
    / "b11-basis-diagnostics.json"
)


def selected_b7_coefficients() -> np.ndarray:
    projected = json.loads(PROJECTED.read_text())
    selected_name = projected["selected_for_expanded_hard_validation"][0]
    row = next(
        checkpoint
        for family in projected["candidates"]
        for checkpoint in family["full_training_contenders"]
        if checkpoint["name"] == selected_name
    )
    return np.asarray([*row["phase_coefficients"], 0.0, 0.0, 0.0, 0.0])


def probes() -> dict[str, np.ndarray]:
    result = {
        "clear": np.zeros(11),
        "best_b7_projected_extended": selected_b7_coefficients(),
    }
    for index, noll in enumerate(range(12, 16), start=7):
        value = np.zeros(11)
        value[index] = 1.0
        result[f"pure_noll_{noll}_1.0"] = value
    mixed = selected_b7_coefficients()
    mixed[7:] = [0.35, -0.35, 0.35, -0.35]
    norm = float(np.linalg.norm(mixed))
    if norm >= 2.5:
        mixed *= 2.5 * 0.999 / norm
    result["b7_plus_mixed_fourth_order"] = mixed
    return result


def circular_quantization_error(phase: np.ndarray, levels: int) -> float:
    wrapped = np.mod(phase, 2.0 * np.pi)
    quantized = np.round(wrapped / (2.0 * np.pi) * levels) % levels
    quantized = quantized * (2.0 * np.pi / levels)
    error = np.angle(np.exp(1j * (quantized - wrapped)))
    return float(np.sqrt(np.mean(np.square(error))))


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"B11 basis diagnostic already exists: {OUTPUT}")
    axis = np.linspace(-1.0, 1.0, 401, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius_squared = xx * xx + yy * yy
    inside = radius_squared <= 1.0
    inner = radius_squared <= 0.95**2
    grid = np.stack([yy, xx], axis=-1)
    basis = np.asarray(zernike_basis_b11(jnp.asarray(grid), 1.0))
    disk_basis = basis[:, inside]
    gram = disk_basis @ disk_basis.T / disk_basis.shape[1]
    rows = []
    for name, coefficients in probes().items():
        parameters = unconstrained_from_coefficients_b11(coefficients)
        support = {
            str(float(depth)): float(
                psf_support_energy_fraction_b11(
                    jnp.asarray(parameters), jnp.asarray(depth)
                )
            )
            for depth in DEPTHS
        }
        phase = np.einsum("m,mhw->hw", coefficients, basis)
        spacing = float(axis[1] - axis[0])
        gradient_y, gradient_x = np.gradient(phase, spacing, spacing)
        gradient_magnitude = np.sqrt(
            gradient_x * gradient_x + gradient_y * gradient_y
        )
        rows.append(
            {
                "name": name,
                "coefficients": coefficients.tolist(),
                "phase_rms_radians": float(
                    np.sqrt(np.mean(np.square(phase[inside])))
                ),
                "phase_peak_to_valley_radians": float(np.ptp(phase[inside])),
                "inner_phase_gradient_rms_radians_per_radius": float(
                    np.sqrt(np.mean(np.square(gradient_magnitude[inner])))
                ),
                "inner_phase_gradient_max_radians_per_radius": float(
                    np.max(gradient_magnitude[inner])
                ),
                "support_energy_fraction_by_depth": support,
                "minimum_support_energy_fraction": min(support.values()),
                "wrapped_phase_quantization_rms_error_radians": {
                    "8_levels": circular_quantization_error(phase[inside], 8),
                    "16_levels": circular_quantization_error(phase[inside], 16),
                },
            }
        )
        print(
            json.dumps(
                {"probe": name, "minimum_support": min(support.values())}
            ),
            flush=True,
        )
    report = {
        "status": "passed_b11_basis_diagnostics",
        "test_accessed": False,
        "basis_noll_indices": list(B11_NOLL_INDICES),
        "normalization": "continuous unit-disk RMS-one Noll convention",
        "sample_grid_size": 401,
        "sampled_gram_matrix": gram.tolist(),
        "maximum_diagonal_error": float(
            np.max(np.abs(np.diag(gram) - 1.0))
        ),
        "maximum_off_diagonal_absolute": float(
            np.max(np.abs(gram - np.diag(np.diag(gram))))
        ),
        "minimum_required_support_energy_fraction": 0.995,
        "probes": rows,
    }
    report["minimum_observed_support_energy_fraction"] = min(
        row["minimum_support_energy_fraction"] for row in rows
    )
    report["passed"] = bool(
        report["maximum_diagonal_error"] <= 0.012
        and report["maximum_off_diagonal_absolute"] <= 0.012
        and report["minimum_observed_support_energy_fraction"] >= 0.995
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "maximum_diagonal_error": report["maximum_diagonal_error"],
                "maximum_off_diagonal_absolute": report[
                    "maximum_off_diagonal_absolute"
                ],
                "minimum_support": report[
                    "minimum_observed_support_energy_fraction"
                ],
                "passed": report["passed"],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("B11 basis convention or support gate failed")


if __name__ == "__main__":
    main()
