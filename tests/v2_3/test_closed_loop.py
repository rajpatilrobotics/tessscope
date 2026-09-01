"""Pure objective and batch tests for the v2.3 feedback loop."""

import jax.numpy as jnp
import numpy as np
import pytest

from tessscope.v2.optimization.served import collect_patches
from tessscope.v2_3.closed_loop import (
    OBJECTIVE_PROFILES,
    ClosedLoopWeights,
    loss_from_terms,
    materialize_closed_loop_batch,
)


def test_closed_loop_batch_has_independent_exposure_noise() -> None:
    patches = collect_patches("training", 3, seed=53)
    batch = materialize_closed_loop_batch(patches, noise_seed=4100)
    assert batch.objects.shape == (3, 256, 256)
    assert batch.first_noise_standard_normal.shape == (3, 7, 256, 256)
    assert batch.second_noise_standard_normal.shape == (1, 7, 256, 256)
    assert not np.array_equal(
        batch.first_noise_standard_normal[2],
        batch.second_noise_standard_normal[0],
    )


def test_loss_terms_use_bounded_action_residual_and_fixed_exposure_cost() -> None:
    weights = ClosedLoopWeights(first_segmentation=0.35)
    depths = jnp.asarray([-2.0, 0.0, 2.0])
    action = jnp.asarray([1.0, 0.0, -1.0])
    value, (residual, residual_squared, action_squared) = loss_from_terms(
        jnp.asarray(1.0),
        jnp.asarray(0.8),
        action,
        depths,
        weights,
    )
    assert residual == pytest.approx([-1.0, 0.0, 1.0])
    assert residual_squared == pytest.approx(2.0 / 108.0)
    assert action_squared == pytest.approx(2.0 / 108.0)
    expected = 0.35 + 0.8 + 0.10 * 2.0 / 108.0 + 0.01 * 2.0 / 108.0 + 0.01
    assert value == pytest.approx(expected)


def test_profiles_match_preregistered_single_weight_changes() -> None:
    assert OBJECTIVE_PROFILES["balanced"].first_segmentation == 0.35
    assert OBJECTIVE_PROFILES["first_heavy"].first_segmentation == 0.75
    assert (
        OBJECTIVE_PROFILES["action_heavy"].normalized_residual_depth_squared
        == 0.25
    )
    with pytest.raises(ValueError, match="nonnegative"):
        ClosedLoopWeights(first_segmentation=-0.1)
