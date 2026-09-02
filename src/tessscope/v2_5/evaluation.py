"""Shared soft and physical evaluation helpers for v2.5."""

from __future__ import annotations

import hashlib
import json

import jax.numpy as jnp
import numpy as np

from tessscope.v2.optimization.served import DEPTHS, V2SystemCalibration
from tessscope.v2_1.optics.model import (
    phase_coefficients_b7,
    phase_coefficients_b11,
    psf_support_energy_fraction_b7,
    psf_support_energy_fraction_b11,
)
from tessscope.v2_3.closed_loop import closed_loop_value
from tessscope.v2_5.constrained import ZERO_WEIGHTS


def parameter_sha256(parameters: list[float] | np.ndarray) -> str:
    payload = json.dumps(
        np.asarray(parameters).tolist(), separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def average_closed_loop_forward(
    services,
    parameters: np.ndarray,
    batches,
    calibration: V2SystemCalibration,
    *,
    gradient_mode: str = "exact",
) -> dict[str, float]:
    """Average transparent closed-loop terms without requesting a VJP."""
    rows = []
    for batch in batches:
        _, diagnostics = closed_loop_value(
            *services,
            np.asarray(parameters, dtype=np.float32),
            batch,
            calibration,
            ZERO_WEIGHTS,
            gradient_mode=gradient_mode,
        )
        rows.append(diagnostics)
    return {
        key: float(np.mean([row[key] for row in rows])) for key in rows[0]
    }


def physical_diagnostics(parameters: np.ndarray, *, basis_size: int) -> dict:
    """Evaluate the unchanged RMS and large-support gates for B7 or B11."""
    value = np.asarray(parameters, dtype=np.float32)
    if basis_size == 7:
        coefficients = np.asarray(phase_coefficients_b7(jnp.asarray(value)))
        support_function = psf_support_energy_fraction_b7
    elif basis_size == 11:
        coefficients = np.asarray(phase_coefficients_b11(jnp.asarray(value)))
        support_function = psf_support_energy_fraction_b11
    else:
        raise ValueError("Only B7 and B11 are registered for v2.5")
    support = {
        str(float(depth)): float(
            support_function(jnp.asarray(value), jnp.asarray(depth))
        )
        for depth in DEPTHS
    }
    return {
        "phase_coefficients": coefficients.tolist(),
        "physical_coefficient_norm_radians": float(np.linalg.norm(coefficients)),
        "minimum_support_energy_fraction": min(support.values()),
        "support_energy_fraction_by_depth": support,
    }
