from pathlib import Path

import numpy as np
import pytest

from tessscope.observer.gate0 import run_parity
from tessscope.observer.instanseg import canonicalize_labels, normalize_release_input


def test_release_normalization_is_not_clipped() -> None:
    image = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    normalized = normalize_release_input(image)
    assert normalized.min() < 0
    assert normalized.max() > 1


def test_canonical_labels_ignore_arbitrary_ids() -> None:
    left = np.array([[0, 9, 9], [4, 4, 0]])
    right = np.array([[0, 2, 2], [7, 7, 0]])
    assert np.array_equal(canonicalize_labels(left), canonicalize_labels(right))


@pytest.mark.skipif(
    not Path("artifacts/external/instanseg/model-v0.1.2/instanseg.pt").exists(),
    reason="official model bundle has not been downloaded",
)
def test_official_release_and_raw_head_parity() -> None:
    report = run_parity()
    assert report["passed"] is True
    assert report["official_pixel_exact"] is True
    assert report["reconstructed_pixel_exact_after_deterministic_relabel"] is True
    assert report["reconstructed_foreground_iou"] == 1.0
    assert report["raw_head_shape"] == [1, 5, 256, 256]
