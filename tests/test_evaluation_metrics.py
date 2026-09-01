import numpy as np
import pytest

from tessscope.evaluation.metrics import (
    grouped_bootstrap_difference,
    instance_metrics,
    resample_labels,
    valid_region_labels,
)


def test_panoptic_quality_is_one_for_relabelled_identical_instances() -> None:
    target = np.asarray([[0, 1, 1], [0, 2, 2], [0, 0, 0]])
    prediction = np.asarray([[0, 9, 9], [0, 4, 4], [0, 0, 0]])
    metrics = instance_metrics(target, prediction)
    assert metrics.panoptic_quality == pytest.approx(1.0)
    assert metrics.true_positives == 2
    assert metrics.absolute_count_error == 0
    assert metrics.percentage_count_error == pytest.approx(0.0)
    assert metrics.foreground_dice == pytest.approx(1.0)


def test_panoptic_quality_penalizes_unmatched_instances() -> None:
    target = np.asarray([[1, 1, 0], [0, 0, 2], [0, 0, 2]])
    prediction = np.asarray([[7, 7, 0], [0, 0, 0], [0, 0, 0]])
    metrics = instance_metrics(target, prediction)
    assert metrics.segmentation_quality == pytest.approx(1.0)
    assert metrics.recognition_quality == pytest.approx(2 / 3)
    assert metrics.panoptic_quality == pytest.approx(2 / 3)
    assert metrics.false_negatives == 1


def test_label_resampling_never_introduces_fractional_ids() -> None:
    labels = np.asarray([[0, 3], [8, 8]], dtype=np.int32)
    resized = resample_labels(labels, (5, 7))
    assert resized.shape == (5, 7)
    assert set(np.unique(resized)) <= {0, 3, 8}


def test_grouped_bootstrap_resamples_sources_not_rows() -> None:
    source_ids = np.asarray(["a", "a", "b", "b"])
    candidate = np.asarray([2.0, 4.0, 10.0, 10.0])
    reference = np.asarray([1.0, 1.0, 8.0, 8.0])
    report = grouped_bootstrap_difference(
        source_ids, candidate, reference, replicates=100, seed=4
    )
    assert report["source_count"] == 2
    assert report["mean_difference"] == pytest.approx(2.0)


def test_valid_region_uses_complete_targets_and_prediction_centroids() -> None:
    target = np.zeros((8, 8), dtype=np.int32)
    target[0:3, 3:5] = 1
    target[3:5, 3:5] = 2
    prediction = np.zeros((8, 8), dtype=np.int32)
    prediction[1:5, 2:3] = 7
    prediction[3:5, 4:5] = 9
    valid_target, valid_prediction = valid_region_labels(target, prediction, margin=2)
    assert set(np.unique(valid_target)) == {0, 2}
    assert set(np.unique(valid_prediction)) == {0, 7, 9}
