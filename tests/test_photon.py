import numpy as np
import pytest

from tessscope.optics.photon import (
    expected_counts,
    normalized_counts,
    sample_poisson_rate,
)


def test_expected_and_normalized_counts_round_trip() -> None:
    rate = np.asarray([[0.0, 0.25, 1.0]], dtype=np.float32)
    counts = expected_counts(rate, 100)
    assert np.array_equal(counts, np.asarray([[0.0, 25.0, 100.0]]))
    assert np.array_equal(normalized_counts(counts, 100), rate)


def test_poisson_rate_is_keyed_and_reproducible() -> None:
    rate = np.full((32, 32), 0.4, dtype=np.float32)
    first = sample_poisson_rate(
        rate,
        50,
        source_image_id="example",
        depth_um=-6,
        replicate=0,
    )
    repeated = sample_poisson_rate(
        rate,
        50,
        source_image_id="example",
        depth_um=-6,
        replicate=0,
    )
    different = sample_poisson_rate(
        rate,
        50,
        source_image_id="example",
        depth_um=-6,
        replicate=1,
    )
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert first.mean() == pytest.approx(0.4, abs=0.02)


def test_negative_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        expected_counts(np.asarray([-0.1], dtype=np.float32), 100)
