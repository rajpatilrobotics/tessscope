import numpy as np
import pytest

from tessscope.observer.calibration import (
    TransformScore,
    UnitIntervalHistogram,
    choose_transform,
    load_frozen_transform,
    transform_candidates,
)


def test_streaming_histogram_quantiles_are_deterministic() -> None:
    histogram = UnitIntervalHistogram(bins=101)
    histogram.update(np.asarray([0.0, 0.25, 0.5]))
    histogram.update(np.asarray([0.75, 1.0]))
    assert histogram.percentile(0) == pytest.approx(0.0)
    assert histogram.percentile(50) == pytest.approx(0.5)
    assert histogram.percentile(100) == pytest.approx(1.0)


def test_transform_candidates_share_global_bounds() -> None:
    candidates = transform_candidates(0.1, 0.6)
    assert candidates["global_affine"].offset == pytest.approx(0.1)
    assert candidates["global_asinh"].scale == pytest.approx(0.5)


def test_affine_wins_an_exact_validation_tie() -> None:
    scores = [
        TransformScore("global_asinh", 0.5, 0.3, 0.7),
        TransformScore("global_affine", 0.5, 0.3, 0.7),
    ]
    assert choose_transform(scores, standard_focus_pq=0.52).name == "global_affine"


def test_asinh_is_used_only_when_affine_fails_focus_gate() -> None:
    scores = [
        TransformScore("global_asinh", 0.5, 0.3, 0.50),
        TransformScore("global_affine", 0.6, 0.4, 0.45),
    ]
    assert choose_transform(scores, standard_focus_pq=0.52).name == "global_asinh"


def test_tracked_frozen_transform_is_loadable() -> None:
    transform = load_frozen_transform()
    assert transform.mode == "affine"
    assert transform.scale > 0
