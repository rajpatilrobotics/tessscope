import numpy as np
import pytest

from tessscope.evaluation.route import (
    evaluation_condition_counts,
    poisson_sensor_batch,
    project_test_runtime,
)


def test_locked_evaluation_condition_count() -> None:
    counts = evaluation_condition_counts()
    assert counts.deterministic_per_source_design == 7
    assert counts.poisson_per_source_design == 16
    assert counts.source_design_pairs == 205
    assert counts.total_observer_images == 4715


def test_runtime_projection_requires_25_percent_contingency() -> None:
    counts = evaluation_condition_counts(source_count=2, design_count=1)
    projection = project_test_runtime(
        10.0,
        available_seconds=25.0,
        reserve_fraction=0.25,
        counts=counts,
    )
    assert projection.projected_seconds == pytest.approx(20.0)
    assert projection.contingency_seconds == pytest.approx(5.0)
    assert projection.passed


def test_runtime_projection_fails_when_contingency_does_not_fit() -> None:
    counts = evaluation_condition_counts(source_count=2, design_count=1)
    projection = project_test_runtime(
        10.0,
        available_seconds=24.9,
        reserve_fraction=0.25,
        counts=counts,
    )
    assert not projection.passed


def test_poisson_batch_is_complete_and_reproducible() -> None:
    sensor = np.full((3, 4, 5), 0.5, dtype=np.float32)
    depths = np.asarray([-6, 0, 6], dtype=np.float32)
    first, conditions = poisson_sensor_batch(
        sensor,
        source_image_id="source-a",
        depths_um=depths,
        photon_levels=(50,),
        replicates=2,
    )
    second, _ = poisson_sensor_batch(
        sensor,
        source_image_id="source-a",
        depths_um=depths,
        photon_levels=(50,),
        replicates=2,
    )
    assert first.shape == (4, 4, 5)
    assert len(conditions) == 4
    np.testing.assert_array_equal(first, second)
