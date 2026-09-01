"""Generate, soft-screen, and physically gate the pre-registered B7 frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import (
    DEPTHS,
    V2SystemCalibration,
    collect_patches,
    joint_value,
    materialize_joint_batch,
)
from tessscope.v2_1.optics.model import (
    psf_support_energy_fraction_b7,
    unconstrained_from_coefficients_b7,
    zernike_basis_b7,
)
from tessscope.v2_2.frontier import (
    generate_piecewise_family,
    select_hard_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEPARATE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_1"
    / "optimization"
    / "b7-separate-baselines.json"
)
FROZEN_JOINT = PROJECT_ROOT / "configs" / "v2_2" / "frozen-joint-candidate.json"
CONTRACT = PROJECT_ROOT / "configs" / "v2_2" / "contract.yaml"
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "screening"
    / "piecewise-soft-frontier.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optics-url", default="http://127.0.0.1:8404")
    parser.add_argument("--autofocus-url", default="http://127.0.0.1:8403")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8402")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circular_quantization_error(phase: np.ndarray, levels: int) -> float:
    wrapped = np.mod(phase, 2.0 * np.pi)
    quantized = np.round(wrapped / (2.0 * np.pi) * levels) % levels
    quantized = quantized * (2.0 * np.pi / levels)
    error = np.angle(np.exp(1j * (quantized - wrapped)))
    return float(np.sqrt(np.mean(np.square(error))))


def physical_diagnostics(point: dict, basis: np.ndarray, inside: np.ndarray) -> dict:
    coefficients = np.asarray(point["phase_coefficients"], dtype=np.float64)
    parameters = unconstrained_from_coefficients_b7(coefficients)
    support = {
        str(float(depth)): float(
            psf_support_energy_fraction_b7(
                jnp.asarray(parameters), jnp.asarray(depth)
            )
        )
        for depth in DEPTHS
    }
    phase = np.einsum("m,mhw->hw", coefficients, basis)
    return {
        "name": point["name"],
        "phase_rms_radians": float(np.linalg.norm(coefficients)),
        "minimum_support_energy_fraction": min(support.values()),
        "support_energy_fraction_by_depth": support,
        "wrapped_phase_quantization_rms_error_radians": {
            "8_levels": circular_quantization_error(phase[inside], 8),
            "16_levels": circular_quantization_error(phase[inside], 16),
        },
        "passed": bool(
            np.linalg.norm(coefficients) < 2.5 and min(support.values()) >= 0.995
        ),
    }


def main() -> None:
    args = parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"V2.2 soft-frontier artifact already exists: {OUTPUT}")
    separate = json.loads(SEPARATE.read_text())
    frozen_joint = json.loads(FROZEN_JOINT.read_text())
    family = generate_piecewise_family(
        np.asarray(separate["segmentation_only"]["final_phase_coefficients"]),
        np.asarray(separate["focus_only"]["final_phase_coefficients"]),
    )
    services = (
        Tesseract.from_url(args.optics_url, timeout=180),
        Tesseract.from_url(args.autofocus_url, timeout=180),
        Tesseract.from_url(args.observer_url, timeout=180),
    )
    for service in services:
        service.health()
    calibration = V2SystemCalibration.load()
    patches = collect_patches("validation", 12, seed=53)
    batches = [
        materialize_joint_batch(patches[index : index + 3], noise_seed=None)
        for index in range(0, len(patches), 3)
    ]
    soft_rows = []
    for point in family:
        parameters = unconstrained_from_coefficients_b7(
            np.asarray(point["phase_coefficients"])
        )
        batch_rows = []
        for batch in batches:
            _, branches = joint_value(
                *services,
                parameters,
                batch,
                calibration,
                JointObjectiveWeights(),
            )
            batch_rows.append(branches)
        row = {
            **point,
            "parameters": parameters.tolist(),
            **{
                key: float(np.mean([value[key] for value in batch_rows]))
                for key in batch_rows[0]
            },
        }
        soft_rows.append(row)
        print(
            json.dumps(
                {
                    "name": row["name"],
                    "segmentation_loss": row["segmentation_loss"],
                    "focus_mse": row["focus_mse"],
                }
            ),
            flush=True,
        )
    selected, selection_reasons = select_hard_candidates(
        soft_rows,
        joint_segmentation_loss=float(
            frozen_joint["frozen_v2_1_metrics"][
                "soft_validation_segmentation_loss"
            ]
        ),
        joint_focus_mse=float(
            frozen_joint["frozen_v2_1_metrics"]["soft_validation_focus_mse"]
        ),
    )
    by_name = {row["name"]: row for row in soft_rows}
    axis = np.linspace(-1.0, 1.0, 401, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    inside = xx * xx + yy * yy <= 1.0
    basis = np.asarray(
        zernike_basis_b7(jnp.asarray(np.stack([yy, xx], axis=-1)), 1.0)
    )
    diagnostics = [
        physical_diagnostics(by_name[name], basis, inside) for name in selected
    ]
    report = {
        "status": "complete_soft_screen_validation_only",
        "test_accessed": False,
        "protocol_sha256": sha256(CONTRACT),
        "frozen_joint_sha256": sha256(FROZEN_JOINT),
        "source_separate_sha256": sha256(SEPARATE),
        "raw_alias_count": 101,
        "unique_point_count": len(soft_rows),
        "selected_for_hard_validation": selected,
        "selected_point_count": len(selected),
        "selection_reasons": selection_reasons,
        "all_selected_pass_physical_gates": all(row["passed"] for row in diagnostics),
        "physical_diagnostics": diagnostics,
        "points": soft_rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "unique_points": len(soft_rows),
                "selected_points": len(selected),
                "physical_gates_passed": report["all_selected_pass_physical_gates"],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    if not report["all_selected_pass_physical_gates"]:
        raise SystemExit("A selected v2.2 frontier point failed a physical gate")


if __name__ == "__main__":
    main()
