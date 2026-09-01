"""Joint objective scaling and branch-weight checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tessscope.v2.optimization.objective import JointObjectiveWeights


def test_joint_objective_gradient_is_weighted_branch_sum() -> None:
    weights = JointObjectiveWeights(segmentation=2.0, focus=3.0, maximum_depth_um=6.0)

    gradient = jax.grad(
        lambda value: weights.combine(value**2, (value - 1.0) ** 2)
    )(jnp.asarray(0.25))
    expected = 2.0 * (2.0 * 0.25) + 3.0 / 36.0 * (2.0 * (0.25 - 1.0))

    assert float(gradient) == pytest.approx(expected)


def test_joint_objective_rejects_two_disabled_branches() -> None:
    with pytest.raises(ValueError, match="At least one"):
        JointObjectiveWeights(segmentation=0.0, focus=0.0).validate()
