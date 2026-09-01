"""Record B7 convention, support, smoothness, and quantization diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tessscope.v2.optimization.served import DEPTHS
from tessscope.v2_1.optics.model import (
    B7_NOLL_INDICES,
    psf_support_energy_fraction_b7,
    unconstrained_from_coefficients_b7,
    zernike_basis_b7,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "artifacts" / "runs" / "v2_1" / "gates" / "b7-basis-diagnostics.json"


def project_inside_ball(coefficients: np.ndarray, bound: float = 2.5) -> np.ndarray:
    value = np.asarray(coefficients, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm >= bound:
        value = value * (bound * 0.999 / norm)
    return value


def probe_coefficients() -> dict[str, np.ndarray]:
    refinement = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "v2"
            / "optimization"
            / "pareto-refinement-from-joint.json"
        ).read_text()
    )
    near = next(row for row in refinement["candidates"] if row["name"] == "joint-refine-focus-0.03")
    extended_near = np.asarray([*near["final_phase_coefficients"], 0.0])
    probes = {
        "clear": np.zeros(7),
        "v2_near_miss_extended": extended_near,
    }
    for strength in (0.5, 1.0, 1.5, 2.0, 2.4):
        value = np.zeros(7)
        value[6] = strength
        probes[f"pure_spherical_{strength:.1f}"] = value
    for strength in (-0.5, 0.5):
        value = extended_near.copy()
        value[6] = strength
        probes[f"near_plus_spherical_{strength:+.1f}"] = project_inside_ball(value)
    return probes


def circular_quantization_error(phase: np.ndarray, levels: int) -> float:
    wrapped = np.mod(phase, 2.0 * np.pi)
    quantized = np.round(wrapped / (2.0 * np.pi) * levels) % levels
    quantized = quantized * (2.0 * np.pi / levels)
    circular_error = np.angle(np.exp(1j * (quantized - wrapped)))
    return float(np.sqrt(np.mean(np.square(circular_error))))


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"B7 basis diagnostic already exists: {OUTPUT}")
    axis = np.linspace(-1.0, 1.0, 401, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius_squared = xx * xx + yy * yy
    inside = radius_squared <= 1.0
    inner = radius_squared <= 0.95**2
    grid = np.stack([yy, xx], axis=-1)
    basis = np.asarray(zernike_basis_b7(jnp.asarray(grid), 1.0))
    disk_basis = basis[:, inside]
    gram = disk_basis @ disk_basis.T / disk_basis.shape[1]
    probes = []
    for name, coefficients in probe_coefficients().items():
        parameters = unconstrained_from_coefficients_b7(coefficients)
        support = {
            str(float(depth)): float(
                psf_support_energy_fraction_b7(jnp.asarray(parameters), jnp.asarray(depth))
            )
            for depth in DEPTHS
        }
        phase = np.einsum("m,mhw->hw", coefficients, basis)
        spacing = float(axis[1] - axis[0])
        gradient_y, gradient_x = np.gradient(phase, spacing, spacing)
        gradient_magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
        probes.append(
            {
                "name": name,
                "coefficients": coefficients.tolist(),
                "phase_rms_radians": float(np.sqrt(np.mean(np.square(phase[inside])))),
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
                {
                    "probe": name,
                    "minimum_support": min(support.values()),
                }
            ),
            flush=True,
        )
    report = {
        "status": "passed_b7_basis_diagnostics",
        "test_accessed": False,
        "basis_noll_indices": list(B7_NOLL_INDICES),
        "normalization": "continuous unit-disk RMS-one Noll convention",
        "sample_grid_size": 401,
        "sampled_gram_matrix": gram.tolist(),
        "maximum_diagonal_error": float(np.max(np.abs(np.diag(gram) - 1.0))),
        "maximum_off_diagonal_absolute": float(np.max(np.abs(gram - np.diag(np.diag(gram))))),
        "minimum_required_support_energy_fraction": 0.995,
        "probes": probes,
    }
    report["minimum_observed_support_energy_fraction"] = min(
        row["minimum_support_energy_fraction"] for row in probes
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
                "maximum_off_diagonal_absolute": report["maximum_off_diagonal_absolute"],
                "minimum_support": report["minimum_observed_support_energy_fraction"],
                "passed": report["passed"],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("B7 basis convention or support gate failed")


if __name__ == "__main__":
    main()
