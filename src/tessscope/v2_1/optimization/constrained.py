"""Small exact-Jacobian epsilon-constraint wrapper around SciPy SLSQP."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize


@dataclass(frozen=True)
class BranchEvaluation:
    """Two branch values and exact gradients at one parameter vector."""

    segmentation_loss: float
    focus_mse: float
    segmentation_gradient: np.ndarray
    normalized_focus_gradient: np.ndarray


class CachedEpsilonConstraint:
    """Expose one expensive exact branch evaluation to all SLSQP callbacks."""

    def __init__(
        self,
        evaluator: Callable[[np.ndarray], BranchEvaluation],
        segmentation_limit: float,
        *,
        focus_normalizer_um2: float = 36.0,
    ) -> None:
        if segmentation_limit <= 0 or focus_normalizer_um2 <= 0:
            raise ValueError("Constraint limit and focus normalizer must be positive")
        self.evaluator = evaluator
        self.segmentation_limit = float(segmentation_limit)
        self.focus_normalizer_um2 = float(focus_normalizer_um2)
        self.cache_misses = 0
        self._parameters: np.ndarray | None = None
        self._evaluation: BranchEvaluation | None = None

    def evaluate(self, parameters: np.ndarray) -> BranchEvaluation:
        value = np.asarray(parameters, dtype=np.float64)
        if self._parameters is None or not np.array_equal(value, self._parameters):
            evaluation = self.evaluator(value.copy())
            if evaluation.segmentation_gradient.shape != value.shape:
                raise ValueError("Segmentation gradient shape does not match parameters")
            if evaluation.normalized_focus_gradient.shape != value.shape:
                raise ValueError("Focus gradient shape does not match parameters")
            self._parameters = value.copy()
            self._evaluation = evaluation
            self.cache_misses += 1
        if self._evaluation is None:
            raise RuntimeError("Constraint evaluation cache was not populated")
        return self._evaluation

    def objective(self, parameters: np.ndarray) -> float:
        return self.evaluate(parameters).focus_mse / self.focus_normalizer_um2

    def objective_jacobian(self, parameters: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.evaluate(parameters).normalized_focus_gradient,
            dtype=np.float64,
        )

    def constraint(self, parameters: np.ndarray) -> float:
        return self.segmentation_limit - self.evaluate(parameters).segmentation_loss

    def constraint_jacobian(self, parameters: np.ndarray) -> np.ndarray:
        return -np.asarray(
            self.evaluate(parameters).segmentation_gradient,
            dtype=np.float64,
        )


def solve_slsqp(
    problem: CachedEpsilonConstraint,
    start_parameters: np.ndarray,
    *,
    trust_radius: float,
    maximum_iterations: int,
    ftol: float = 1e-5,
) -> tuple[OptimizeResult, list[dict]]:
    """Minimize normalized focus under one exact segmentation-loss constraint."""
    start = np.asarray(start_parameters, dtype=np.float64)
    if start.ndim != 1 or not np.isfinite(start).all():
        raise ValueError("Start parameters must be a finite vector")
    if trust_radius <= 0 or maximum_iterations <= 0 or ftol <= 0:
        raise ValueError("SLSQP budget values must be positive")
    lower = np.maximum(-3.0, start - trust_radius)
    upper = np.minimum(3.0, start + trust_radius)
    trace = []

    def callback(parameters: np.ndarray) -> None:
        evaluation = problem.evaluate(parameters)
        trace.append(
            {
                "iteration": len(trace) + 1,
                "parameters": np.asarray(parameters).tolist(),
                "segmentation_loss": evaluation.segmentation_loss,
                "focus_mse": evaluation.focus_mse,
                "constraint_slack": (
                    problem.segmentation_limit - evaluation.segmentation_loss
                ),
                "segmentation_gradient_norm": float(
                    np.linalg.norm(evaluation.segmentation_gradient)
                ),
                "normalized_focus_gradient_norm": float(
                    np.linalg.norm(evaluation.normalized_focus_gradient)
                ),
            }
        )

    result = minimize(
        problem.objective,
        start,
        method="SLSQP",
        jac=problem.objective_jacobian,
        bounds=Bounds(lower, upper),
        constraints={
            "type": "ineq",
            "fun": problem.constraint,
            "jac": problem.constraint_jacobian,
        },
        callback=callback,
        options={
            "ftol": ftol,
            "maxiter": maximum_iterations,
            "disp": False,
        },
    )
    return result, trace


def is_promotion_eligible(
    training_constraint_slack: float,
    validation_segmentation_loss: float,
    validation_segmentation_limit: float,
    *,
    slack_tolerance: float = 1e-4,
) -> bool:
    """Require finite training feasibility and held-out segmentation eligibility."""
    values = np.asarray(
        [
            training_constraint_slack,
            validation_segmentation_loss,
            validation_segmentation_limit,
            slack_tolerance,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or slack_tolerance < 0:
        return False
    return bool(
        training_constraint_slack >= -slack_tolerance
        and validation_segmentation_loss <= validation_segmentation_limit
    )
