"""Pure data, parameter, and metric tests for the v2.6 two-mask graph."""

import jax.numpy as jnp
import numpy as np
import pytest

from tessscope.v2_6.data import materialize_two_mask_batches
from tessscope.v2_6.two_mask import (
    TwoMaskWeights,
    join_two_mask_parameters,
    split_two_mask_parameters,
)


def test_join_and_split_preserve_sensing_capture_order() -> None:
    sensing = np.arange(7, dtype=np.float32)
    capture = np.arange(7, 14, dtype=np.float32)
    joined = join_two_mask_parameters(sensing, capture)
    observed_sensing, observed_capture = split_two_mask_parameters(jnp.asarray(joined))
    assert joined.shape == (14,)
    assert observed_sensing == pytest.approx(sensing)
    assert observed_capture == pytest.approx(capture)


def test_parameter_shapes_and_negative_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="two seven-value"):
        join_two_mask_parameters(np.zeros(6), np.zeros(7))
    with pytest.raises(ValueError, match="Expected 14"):
        split_two_mask_parameters(jnp.zeros(13))
    with pytest.raises(ValueError, match="nonnegative"):
        TwoMaskWeights(first_segmentation=-1.0)


@pytest.mark.parametrize("partition", ["optimization", "development", "confirmation"])
def test_two_mask_batches_use_frozen_training_only_wells(partition: str) -> None:
    batches = materialize_two_mask_batches(partition)
    assert len(batches) == 4
    assert len({patch_id.split("_")[0] for batch in batches for patch_id in batch.patch_ids}) == 12
    assert all(batch.objects.shape == (3, 256, 256) for batch in batches)
    assert all(batch.first_noise_standard_normal.shape == (3, 7, 256, 256) for batch in batches)
    assert all(batch.second_noise_standard_normal.shape == (1, 7, 256, 256) for batch in batches)
