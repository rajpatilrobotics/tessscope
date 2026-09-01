"""Tests for exact-Jacobian constrained continuation helpers."""

import numpy as np
import pytest

from tessscope.v2_1.optimization.constrained import (
    BranchEvaluation,
    CachedEpsilonConstraint,
    solve_slsqp,
)


def test_cached_problem_shares_one_branch_evaluation() -> None:
    calls = 0

    def evaluator(parameters: np.ndarray) -> BranchEvaluation:
        nonlocal calls
        calls += 1
        return BranchEvaluation(
            segmentation_loss=float(parameters[0] ** 2),
            focus_mse=float((parameters[0] - 2.0) ** 2),
            segmentation_gradient=np.asarray([2.0 * parameters[0]]),
            normalized_focus_gradient=np.asarray([(parameters[0] - 2.0) / 18.0]),
        )

    problem = CachedEpsilonConstraint(evaluator, segmentation_limit=0.25)
    point = np.asarray([0.1])
    problem.objective(point)
    problem.objective_jacobian(point)
    problem.constraint(point)
    problem.constraint_jacobian(point)
    assert calls == 1
    assert problem.constraint(point) == pytest.approx(0.24)


def test_slsqp_reaches_focus_boundary_under_segmentation_constraint() -> None:
    def evaluator(parameters: np.ndarray) -> BranchEvaluation:
        return BranchEvaluation(
            segmentation_loss=float(parameters[0] ** 2),
            focus_mse=float((parameters[0] - 2.0) ** 2),
            segmentation_gradient=np.asarray([2.0 * parameters[0]]),
            normalized_focus_gradient=np.asarray([(parameters[0] - 2.0) / 18.0]),
        )

    problem = CachedEpsilonConstraint(evaluator, segmentation_limit=0.25)
    result, trace = solve_slsqp(
        problem,
        np.asarray([0.0]),
        trust_radius=1.0,
        maximum_iterations=30,
    )
    assert result.success
    assert result.x[0] == pytest.approx(0.5, abs=2e-4)
    assert problem.constraint(result.x) >= -1e-5
    assert trace
