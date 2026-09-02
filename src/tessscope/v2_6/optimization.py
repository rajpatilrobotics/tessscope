"""Exact feasibility-restoration and constrained objectives for two masks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration
from tessscope.v2_3.closed_loop import ClosedLoopBatch
from tessscope.v2_5.constrained import augmented_lagrangian_terms, scaled_violations
from tessscope.v2_6.two_mask import FINAL_WEIGHTS, _diagnostics, _graph


def _averaged_auxiliary(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: jax.Array,
    batches: tuple[ClosedLoopBatch, ...] | list[ClosedLoopBatch],
    calibration: V2SystemCalibration,
) -> tuple[jax.Array, ...]:
    rows = [
        _graph(
            optics,
            autofocus,
            observer,
            parameters,
            batch,
            calibration,
            FINAL_WEIGHTS,
            "exact",
        )[1]
        for batch in batches
    ]
    if not rows:
        raise ValueError("At least one two-mask batch is required")
    return tuple(
        jnp.mean(jnp.stack([row[index] for row in rows]))
        for index in range(len(rows[0]))
    )


def feasibility_value_and_gradient(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batches: tuple[ClosedLoopBatch, ...] | list[ClosedLoopBatch],
    calibration: V2SystemCalibration,
    *,
    first_limit: float,
    residual_limit: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Differentiate squared positive scaled violations over all four batches."""

    def objective(value: jax.Array):
        auxiliary = _averaged_auxiliary(
            optics, autofocus, observer, value, batches, calibration
        )
        violations = jnp.stack(
            [
                (auxiliary[0] - first_limit) / 0.008,
                (auxiliary[4] - residual_limit) / 0.02,
            ]
        )
        feasibility = jnp.sum(jnp.square(jnp.maximum(violations, 0.0)))
        return feasibility, (auxiliary, violations)

    (value, (auxiliary, violations)), gradient = jax.value_and_grad(
        objective, has_aux=True
    )(jnp.asarray(parameters, dtype=jnp.float32))
    diagnostics = _diagnostics(auxiliary)
    diagnostics.update(
        {
            "first_scaled_violation": float(violations[0]),
            "residual_scaled_violation": float(violations[1]),
        }
    )
    return float(value), np.asarray(gradient, dtype=np.float32), diagnostics


def constrained_value_and_gradient(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batches: tuple[ClosedLoopBatch, ...] | list[ClosedLoopBatch],
    calibration: V2SystemCalibration,
    *,
    first_limit: float,
    residual_limit: float,
    duals: np.ndarray,
    penalty: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Differentiate final loss plus the frozen two-constraint PHR term."""

    def objective(value: jax.Array):
        auxiliary = _averaged_auxiliary(
            optics, autofocus, observer, value, batches, calibration
        )
        augmented, violations = augmented_lagrangian_terms(
            auxiliary[1],
            auxiliary[0],
            auxiliary[4],
            first_limit=first_limit,
            residual_limit=residual_limit,
            duals=jnp.asarray(duals, dtype=jnp.float32),
            penalty=penalty,
        )
        return augmented, (auxiliary, violations)

    (value, (auxiliary, violations)), gradient = jax.value_and_grad(
        objective, has_aux=True
    )(jnp.asarray(parameters, dtype=jnp.float32))
    diagnostics = _diagnostics(auxiliary)
    diagnostics.update(
        {
            "first_scaled_violation": float(violations[0]),
            "residual_scaled_violation": float(violations[1]),
            "first_dual": float(duals[0]),
            "residual_dual": float(duals[1]),
            "penalty": float(penalty),
        }
    )
    return float(value), np.asarray(gradient, dtype=np.float32), diagnostics


def project_two_masks(parameters: np.ndarray) -> tuple[np.ndarray, tuple[bool, bool]]:
    """Apply the frozen 2.4975-radian projection independently to both B7 masks."""
    from tessscope.v2_5.constrained import project_to_physical_radius

    value = np.asarray(parameters, dtype=np.float32)
    if value.shape != (14,):
        raise ValueError("Two-mask projection requires 14 parameters")
    sensing, sensing_changed = project_to_physical_radius(value[:7], basis_size=7)
    capture, capture_changed = project_to_physical_radius(value[7:], basis_size=7)
    return np.concatenate([sensing, capture]), (sensing_changed, capture_changed)


def feasibility_violations(
    metrics: dict,
    *,
    first_limit: float,
    residual_limit: float,
) -> np.ndarray:
    """Expose the unchanged scaled violations for checkpoint decisions."""
    return scaled_violations(
        metrics["first_segmentation_loss"],
        metrics["normalized_residual_depth_squared"],
        first_limit=first_limit,
        residual_limit=residual_limit,
    )
