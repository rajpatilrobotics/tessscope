import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tessscope.optimization.adam import AdamState, adam_update
from tessscope.optimization.objectives import image_fidelity_loss, structural_similarity
from tessscope.optimization.schedule import build_training_schedule


def test_training_schedule_is_reproducible_and_uses_frozen_depth_rule() -> None:
    first = build_training_schedule(steps=4)
    second = build_training_schedule(steps=4)
    assert first == second
    assert first[0].depth_indices == (0, 2, 4)
    assert first[1].depth_indices == (1, 3, 5)
    assert len({patch for batch in first for patch in batch.patches}) == 8


def test_adam_first_step_has_expected_sign() -> None:
    parameters = np.asarray([1.0, -1.0], dtype=np.float32)
    gradient = np.asarray([2.0, -3.0], dtype=np.float32)
    updated, state = adam_update(
        parameters,
        gradient,
        AdamState.zeros(parameters.shape),
        learning_rate=0.1,
    )
    np.testing.assert_allclose(updated, [0.9, -0.9], atol=1e-6)
    assert state.step == 1


def test_image_fidelity_is_zero_for_identical_images_and_differentiable() -> None:
    reference = jnp.linspace(0.0, 1.0, 32 * 32).reshape(1, 1, 32, 32)
    sensor = jnp.broadcast_to(reference, (1, 3, 32, 32))
    phase = jnp.zeros((6,), dtype=jnp.float32)
    loss = image_fidelity_loss(sensor, reference, phase)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    gradient = jax.grad(lambda value: image_fidelity_loss(value, reference, phase))(
        sensor + 0.01
    )
    assert np.isfinite(np.asarray(gradient)).all()


def test_structural_similarity_decreases_after_large_change() -> None:
    reference = jnp.ones((1, 1, 24, 24), dtype=jnp.float32) * 0.5
    changed = reference.at[..., 6:18, 6:18].set(0.0)
    assert float(structural_similarity(reference, reference)) == pytest.approx(
        1.0, abs=1e-5
    )
    assert float(structural_similarity(changed, reference)) < 1.0
