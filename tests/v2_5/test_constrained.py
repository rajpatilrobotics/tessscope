"""Pure tests for the v2.5 constrained methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tessscope.v2_5.constrained import (
    CachedClosedLoopConstraints,
    ConstraintEvaluation,
    augmented_lagrangian_terms,
    project_to_physical_radius,
    scaled_violations,
    solve_slsqp,
    update_duals,
)


def test_augmented_lagrangian_has_exact_active_constraint_gradient() -> None:
    def objective(value):
        result, _ = augmented_lagrangian_terms(
            final_segmentation_loss=value[0],
            first_segmentation_loss=value[1],
            normalized_residual_depth_squared=value[2],
            first_limit=1.10,
            residual_limit=0.06,
            duals=jnp.asarray([0.5, 0.25]),
            penalty=4.0,
        )
        return result

    point = jnp.asarray([1.0, 1.104, 0.065])
    gradient = jax.grad(objective)(point)
    assert gradient[0] == pytest.approx(1.0)
    assert gradient[1] > 0.0
    assert gradient[2] > 0.0


def test_scaled_violations_and_projected_dual_update() -> None:
    violations = scaled_violations(
        1.104,
        0.05,
        first_limit=1.10,
        residual_limit=0.06,
    )
    assert violations == pytest.approx([0.5, -0.5])
    assert update_duals(
        np.asarray([0.0, 1.0]), violations, penalty=4.0
    ) == pytest.approx([2.0, 0.0])


@pytest.mark.parametrize("basis_size", [7, 11])
def test_physical_projection_stays_inside_registered_radius(basis_size: int) -> None:
    parameters = np.full(basis_size, 30.0, dtype=np.float32)
    projected, changed = project_to_physical_radius(
        parameters, basis_size=basis_size
    )
    assert changed
    assert np.isfinite(projected).all()
    reprojection, changed_again = project_to_physical_radius(
        projected, basis_size=basis_size
    )
    assert not changed_again
    assert reprojection == pytest.approx(projected)


def test_two_constraint_slsqp_uses_cached_exact_jacobian() -> None:
    def evaluator(parameters: np.ndarray) -> ConstraintEvaluation:
        x, y = parameters
        return ConstraintEvaluation(
            final_segmentation_loss=(x - 1.0) ** 2 + (y - 1.0) ** 2,
            first_segmentation_loss=x,
            normalized_residual_depth_squared=y,
            final_gradient=np.asarray([2.0 * (x - 1.0), 2.0 * (y - 1.0)]),
            first_gradient=np.asarray([1.0, 0.0]),
            residual_gradient=np.asarray([0.0, 1.0]),
        )

    problem = CachedClosedLoopConstraints(
        evaluator, first_limit=0.4, residual_limit=0.3
    )
    result, trace = solve_slsqp(
        problem,
        np.asarray([0.0, 0.0]),
        trust_radius=2.0,
        maximum_iterations=20,
    )
    assert result.success
    assert result.x == pytest.approx([0.4, 0.3], abs=1e-5)
    assert trace
    assert problem.cache_misses < 30
