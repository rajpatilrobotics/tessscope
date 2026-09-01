from pathlib import Path

import numpy as np
import pytest

from tessscope.data.bbbc039 import (
    ObjectNormalization,
    observer_size,
    padded_centers,
    patch_origins,
    prepare_patch,
    records_for_split,
    select_patch_instances,
)


def test_patch_origins_are_locked() -> None:
    assert patch_origins("training") == ((0, 0), (0, 440), (264, 0), (264, 440))
    assert patch_origins("validation") == ((0, 220), (264, 220))


def test_select_instances_requires_full_valid_containment() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[7:10, 7:10] = 2
    selected = select_patch_instances(labels, key="test", valid_margin=5)
    assert set(np.unique(selected)) == {0, 1}
    assert np.all(selected[7:10, 7:10] == 1)


def test_keyed_cap_is_repeatable() -> None:
    labels = np.arange(1, 41, dtype=np.int32).reshape(5, 8)
    first = select_patch_instances(labels, key="same", valid_margin=0, maximum_instances=8)
    second = select_patch_instances(labels, key="same", valid_margin=0, maximum_instances=8)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) - 1 == 8


def test_observer_size_and_centers() -> None:
    assert observer_size((256, 256)) == (330, 330)
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:4, 6:8] = 1
    centers, valid = padded_centers(labels, maximum_instances=3)
    assert np.allclose(centers[0], [2.5, 6.5])
    assert valid.tolist() == [True, False, False]


@pytest.mark.skipif(
    not Path("data/external/BBBC039/images/images").exists(),
    reason="BBBC039 has not been downloaded",
)
def test_prepare_training_patch_uses_only_allowed_records() -> None:
    record = records_for_split("training")[0]
    patch = prepare_patch(record, patch_origins("training")[0], ObjectNormalization(0, 65535))
    assert record.excluded is False
    assert patch.object_image.shape == (256, 256)
    assert patch.instance_labels.shape == (330, 330)
    assert patch.centers_yx.shape == (32, 2)
    assert patch.valid_objects.sum() <= 32
