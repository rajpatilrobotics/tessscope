"""Exact constrained closed-loop utilities for the preregistered v2.5 run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize
from tesseract_core import Tesseract

from tessscope.v2.optimization.served import V2SystemCalibration
from tessscope.v2_1.optics.model import (
    phase_coefficients_b7,
    phase_coefficients_b11,
    unconstrained_from_coefficients_b7,
    unconstrained_from_coefficients_b11,
)
from tessscope.v2_3.closed_loop import (
    ClosedLoopBatch,
    ClosedLoopWeights,
    _diagnostics,
    _graph,
)

FIRST_CONSTRAINT_SCALE = 0.008
RESIDUAL_CONSTRAINT_SCALE = 0.02
ZERO_WEIGHTS = ClosedLoopWeights(
    first_segmentation=0.0,
    final_segmentation=0.0,
    normalized_residual_depth_squared=0.0,
    normalized_stage_action_squared=0.0,
    fixed_second_exposure_cost=0.0,
)


def scaled_violations(
    first_segmentation_loss: float,
    normalized_residual_depth_squared: float,
    *,
    first_limit: float,
    residual_limit: float,
) -> np.ndarray:
    """Return dimensionless signed inequality violations; feasible values are <= 0."""
    return np.asarray(
        [
            (first_segmentation_loss - first_limit) / FIRST_CONSTRAINT_SCALE,
            (normalized_residual_depth_squared - residual_limit)
            / RESIDUAL_CONSTRAINT_SCALE,
        ],
        dtype=np.float64,
    )


def augmented_lagrangian_terms(
    final_segmentation_loss: jax.Array,
    first_segmentation_loss: jax.Array,
    normalized_residual_depth_squared: jax.Array,
    *,
    first_limit: float,
    residual_limit: float,
    duals: jax.Array,
    penalty: float,
) -> tuple[jax.Array, jax.Array]:
    """Apply the Powell-Hestenes-Rockafellar inequality augmented Lagrangian."""
    if penalty <= 0:
        raise ValueError("Augmented-Lagrangian penalty must be positive")
    dual_values = jnp.asarray(duals, dtype=jnp.float32)
    if dual_values.shape != (2,) or bool(jnp.any(dual_values < 0.0)):
        raise ValueError("Exactly two nonnegative dual values are required")
    violations = jnp.stack(
        [
            (first_segmentation_loss - first_limit) / FIRST_CONSTRAINT_SCALE,
            (normalized_residual_depth_squared - residual_limit)
            / RESIDUAL_CONSTRAINT_SCALE,
        ]
    )
    shifted = jnp.maximum(0.0, dual_values + penalty * violations)
    constraint_term = jnp.sum(
        (jnp.square(shifted) - jnp.square(dual_values)) / (2.0 * penalty)
    )
    return final_segmentation_loss + constraint_term, violations


def closed_loop_augmented_value_and_gradient(
    optics: Tesseract,
    autofocus: Tesseract,
    observer: Tesseract,
    parameters: np.ndarray,
    batch: ClosedLoopBatch,
    calibration: V2SystemCalibration,
    *,
    first_limit: float,
    residual_limit: float,
    duals: np.ndarray,
    penalty: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Differentiate one exact full-loop augmented-Lagrangian training batch."""

    def objective(value: jax.Array):
        _, auxiliary = _graph(
            optics,
            autofocus,
            observer,
            value,
            batch,
            calibration,
            ZERO_WEIGHTS,
            "exact",
        )
        first_loss = auxiliary[0]
        final_loss = auxiliary[1]
        residual_squared = auxiliary[4]
        augmented, violations = augmented_lagrangian_terms(
            final_loss,
            first_loss,
            residual_squared,
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
            "penalty": float(penalty),
            "first_dual": float(duals[0]),
            "residual_dual": float(duals[1]),
        }
    )
    return float(value), np.asarray(gradient), diagnostics


def update_duals(
    duals: np.ndarray,
    violations: np.ndarray,
    *,
    penalty: float,
    maximum: float = 64.0,
) -> np.ndarray:
    """Perform the frozen projected dual-ascent update."""
    value = np.asarray(duals, dtype=np.float64)
    violation = np.asarray(violations, dtype=np.float64)
    if value.shape != (2,) or violation.shape != (2,):
        raise ValueError("Duals and violations must each contain two values")
    if penalty <= 0 or maximum <= 0:
        raise ValueError("Penalty and maximum dual value must be positive")
    return np.clip(value + penalty * violation, 0.0, maximum).astype(np.float32)


def project_to_physical_radius(
    parameters: np.ndarray,
    *,
    basis_size: int,
    radius: float = 2.4975,
) -> tuple[np.ndarray, bool]:
    """Radially project physical coefficients and return unconstrained parameters."""
    value = np.asarray(parameters, dtype=np.float32)
    if basis_size == 7:
        coefficients = np.asarray(phase_coefficients_b7(jnp.asarray(value)))
        inverse = unconstrained_from_coefficients_b7
    elif basis_size == 11:
        coefficients = np.asarray(phase_coefficients_b11(jnp.asarray(value)))
        inverse = unconstrained_from_coefficients_b11
    else:
        raise ValueError("Only the preregistered B7 and B11 bases are supported")
    norm = float(np.linalg.norm(coefficients))
    if norm <= radius:
        return value.copy(), False
    return inverse(coefficients * (radius / norm)), True


@dataclass(frozen=True)
class ConstraintEvaluation:
    """Final objective and two exact constraint values/Jacobians."""

    final_segmentation_loss: float
    first_segmentation_loss: float
    normalized_residual_depth_squared: float
    final_gradient: np.ndarray
    first_gradient: np.ndarray
    residual_gradient: np.ndarray


class CachedClosedLoopConstraints:
    """Share one expensive vector evaluation across all SLSQP callbacks."""

    def __init__(
        self,
        evaluator: Callable[[np.ndarray], ConstraintEvaluation],
        *,
        first_limit: float,
        residual_limit: float,
    ) -> None:
        self.evaluator = evaluator
        self.first_limit = float(first_limit)
        self.residual_limit = float(residual_limit)
        self.cache_misses = 0
        self._parameters: np.ndarray | None = None
        self._evaluation: ConstraintEvaluation | None = None

    def evaluate(self, parameters: np.ndarray) -> ConstraintEvaluation:
        value = np.asarray(parameters, dtype=np.float64)
        if self._parameters is None or not np.array_equal(value, self._parameters):
            result = self.evaluator(value.copy())
            for gradient in (
                result.final_gradient,
                result.first_gradient,
                result.residual_gradient,
            ):
                if gradient.shape != value.shape:
                    raise ValueError("Constraint gradient shape differs from parameters")
            self._parameters = value.copy()
            self._evaluation = result
            self.cache_misses += 1
        if self._evaluation is None:
            raise RuntimeError("Constraint cache was not populated")
        return self._evaluation

    def objective(self, parameters: np.ndarray) -> float:
        return self.evaluate(parameters).final_segmentation_loss

    def objective_jacobian(self, parameters: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate(parameters).final_gradient, dtype=np.float64)

    def constraints(self, parameters: np.ndarray) -> np.ndarray:
        result = self.evaluate(parameters)
        return np.asarray(
            [
                self.first_limit - result.first_segmentation_loss,
                self.residual_limit - result.normalized_residual_depth_squared,
            ]
        )

    def constraint_jacobian(self, parameters: np.ndarray) -> np.ndarray:
        result = self.evaluate(parameters)
        return -np.stack([result.first_gradient, result.residual_gradient])


def solve_slsqp(
    problem: CachedClosedLoopConstraints,
    start_parameters: np.ndarray,
    *,
    trust_radius: float = 0.15,
    maximum_iterations: int = 8,
    ftol: float = 1e-5,
) -> tuple[OptimizeResult, list[dict]]:
    """Run the bounded preregistered exact-Jacobian alternate."""
    start = np.asarray(start_parameters, dtype=np.float64)
    if start.ndim != 1 or not np.isfinite(start).all():
        raise ValueError("Start parameters must be a finite vector")
    if trust_radius <= 0 or maximum_iterations <= 0 or ftol <= 0:
        raise ValueError("SLSQP budget values must be positive")
    trace: list[dict] = []

    def callback(parameters: np.ndarray) -> None:
        evaluation = problem.evaluate(parameters)
        trace.append(
            {
                "iteration": len(trace) + 1,
                "parameters": np.asarray(parameters).tolist(),
                "final_segmentation_loss": evaluation.final_segmentation_loss,
                "first_segmentation_loss": evaluation.first_segmentation_loss,
                "normalized_residual_depth_squared": (
                    evaluation.normalized_residual_depth_squared
                ),
                "constraint_slacks": problem.constraints(parameters).tolist(),
            }
        )

    result = minimize(
        problem.objective,
        start,
        method="SLSQP",
        jac=problem.objective_jacobian,
        bounds=Bounds(start - trust_radius, start + trust_radius),
        constraints={
            "type": "ineq",
            "fun": problem.constraints,
            "jac": problem.constraint_jacobian,
        },
        callback=callback,
        options={"ftol": ftol, "maxiter": maximum_iterations, "disp": False},
    )
    return result, trace
