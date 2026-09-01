"""Locked full-source evaluation route and runtime projection helpers."""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from tessscope.data.bbbc039 import (
    ObjectNormalization,
    decode_instance_mask,
    load_raw_image,
    observer_size,
)
from tessscope.evaluation.metrics import (
    instance_metrics,
    resample_labels,
    valid_region_labels,
)
from tessscope.observer.instanseg import load_frozen_model
from tessscope.observer.loss import (
    DEFAULT_OBSERVER_TRANSFORM,
    ObserverTransform,
    official_hard_labels,
)
from tessscope.optics.model import simulate_sensor
from tessscope.optics.photon import sample_poisson_rate


@dataclass(frozen=True)
class EvaluationConditionCounts:
    """Number of official observer calls in the frozen test matrix."""

    sources: int
    designs: int
    deterministic_per_source_design: int
    poisson_per_source_design: int
    source_design_pairs: int
    total_observer_images: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeProjection:
    """Conservative projected duration plus explicitly reserved contingency."""

    seconds_per_source_design: float
    projected_seconds: float
    reserve_fraction: float
    contingency_seconds: float
    required_seconds: float
    available_seconds: float
    remaining_after_projection_seconds: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def evaluation_condition_counts(
    *,
    source_count: int = 41,
    design_count: int = 5,
    deterministic_depth_count: int = 7,
    poisson_depth_count: int = 2,
    photon_level_count: int = 2,
    poisson_replicates: int = 4,
) -> EvaluationConditionCounts:
    """Count focus, off-focus, and Poisson conditions without double counting."""
    values = (
        source_count,
        design_count,
        deterministic_depth_count,
        poisson_depth_count,
        photon_level_count,
        poisson_replicates,
    )
    if any(value <= 0 for value in values):
        raise ValueError("All evaluation condition counts must be positive")
    poisson_count = poisson_depth_count * photon_level_count * poisson_replicates
    pair_count = source_count * design_count
    return EvaluationConditionCounts(
        sources=source_count,
        designs=design_count,
        deterministic_per_source_design=deterministic_depth_count,
        poisson_per_source_design=poisson_count,
        source_design_pairs=pair_count,
        total_observer_images=pair_count * (deterministic_depth_count + poisson_count),
    )


def project_test_runtime(
    seconds_per_source_design: float,
    *,
    available_seconds: float,
    reserve_fraction: float = 0.25,
    counts: EvaluationConditionCounts | None = None,
) -> RuntimeProjection:
    """Require the test window to cover the projection plus its contingency."""
    if seconds_per_source_design <= 0:
        raise ValueError("Measured route duration must be positive")
    if available_seconds <= 0:
        raise ValueError("Available test window must be positive")
    if not 0 <= reserve_fraction < 1:
        raise ValueError("Reserve fraction must lie in [0, 1)")
    if counts is None:
        counts = evaluation_condition_counts()
    projected = seconds_per_source_design * counts.source_design_pairs
    contingency = reserve_fraction * projected
    required = projected + contingency
    remaining = available_seconds - projected
    return RuntimeProjection(
        seconds_per_source_design=seconds_per_source_design,
        projected_seconds=projected,
        reserve_fraction=reserve_fraction,
        contingency_seconds=contingency,
        required_seconds=required,
        available_seconds=available_seconds,
        remaining_after_projection_seconds=remaining,
        passed=bool(remaining >= contingency),
    )


def full_source_inputs(
    record: Any, normalization: ObjectNormalization
) -> tuple[np.ndarray, np.ndarray]:
    """Load one decontaminated source and its native-grid target."""
    object_image = normalization.apply(load_raw_image(record.image_path))
    target = decode_instance_mask(record.mask_path)
    return object_image, target


def deterministic_sensor(
    phase_parameters: np.ndarray,
    object_image: np.ndarray,
    depths_um: np.ndarray,
) -> np.ndarray:
    """Simulate all deterministic depth planes for one full source image."""
    result = simulate_sensor(
        jnp.asarray(phase_parameters, dtype=jnp.float32),
        jnp.asarray(object_image[None], dtype=jnp.float32),
        jnp.asarray(depths_um, dtype=jnp.float32),
    )
    sensor = np.asarray(result.block_until_ready()[0], dtype=np.float32)
    # FFT roundoff can produce tiny negative values from a physically nonnegative image.
    return np.maximum(sensor, np.float32(0.0))


def poisson_sensor_batch(
    deterministic: np.ndarray,
    *,
    source_image_id: str,
    depths_um: np.ndarray,
    endpoint_depths_um: tuple[float, ...] = (-6.0, 6.0),
    photon_levels: tuple[int, ...] = (50, 200),
    replicates: int = 4,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Build the frozen, keyed Poisson endpoint conditions for one source."""
    sensor = np.asarray(deterministic, dtype=np.float32)
    depths = np.asarray(depths_um, dtype=np.float32)
    if sensor.ndim != 3 or sensor.shape[0] != len(depths):
        raise ValueError("Deterministic sensor must have shape depth,height,width")
    if replicates <= 0:
        raise ValueError("Poisson replicate count must be positive")
    images: list[np.ndarray] = []
    conditions: list[dict[str, float | int | str]] = []
    for depth in endpoint_depths_um:
        matches = np.flatnonzero(np.isclose(depths, depth, rtol=0.0, atol=1e-6))
        if len(matches) != 1:
            raise ValueError(f"Endpoint depth {depth} must occur exactly once")
        rate = sensor[int(matches[0])]
        for photon_level in photon_levels:
            for replicate in range(replicates):
                images.append(
                    sample_poisson_rate(
                        rate,
                        photon_level,
                        source_image_id=source_image_id,
                        depth_um=depth,
                        replicate=replicate,
                    )
                )
                conditions.append(
                    {
                        "kind": "poisson",
                        "depth_um": float(depth),
                        "expected_photons": int(photon_level),
                        "replicate": replicate,
                    }
                )
    return np.stack(images), conditions


def segment_sensor_batch(
    model: torch.nn.Module,
    sensor: np.ndarray,
    observer_shape: tuple[int, int],
    *,
    device: torch.device,
    maximum_batch_size: int = 4,
    transform: ObserverTransform = DEFAULT_OBSERVER_TRANSFORM,
) -> np.ndarray:
    """Run the official frozen hard endpoint in bounded-memory batches."""
    images = np.asarray(sensor, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("Evaluation sensor batch must have shape item,height,width")
    if maximum_batch_size <= 0:
        raise ValueError("Maximum batch size must be positive")
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), maximum_batch_size):
            chunk = torch.from_numpy(images[start : start + maximum_batch_size].copy())
            output = official_hard_labels(
                model,
                chunk[None].to(device),
                observer_shape,
                transform,
            )
            labels.append(output[:, 0].to("cpu").numpy().astype(np.int32))
    return np.concatenate(labels, axis=0)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_validation_route(
    records: list[Any],
    *,
    available_seconds: float = 3600.0,
    design_count: int = 5,
    maximum_batch_size: int = 4,
    transform: ObserverTransform = DEFAULT_OBSERVER_TRANSFORM,
) -> dict[str, Any]:
    """Exercise every condition type and conservatively project the test run."""
    if not records:
        raise ValueError("At least one validation record is required")
    normalization = ObjectNormalization(lower=125.0, upper=1642.0)
    depths = np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32)
    phase_parameters = np.zeros((6,), dtype=np.float32)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_started = time.perf_counter()
    model = load_frozen_model(device)
    model_load_seconds = time.perf_counter() - model_started
    measurements: list[dict[str, Any]] = []

    for record in records:
        total_started = time.perf_counter()
        load_started = time.perf_counter()
        object_image, target = full_source_inputs(record, normalization)
        load_seconds = time.perf_counter() - load_started

        optics_started = time.perf_counter()
        deterministic = deterministic_sensor(phase_parameters, object_image, depths)
        optics_seconds = time.perf_counter() - optics_started

        photon_started = time.perf_counter()
        poisson, poisson_conditions = poisson_sensor_batch(
            deterministic,
            source_image_id=record.image_id,
            depths_um=depths,
        )
        photon_seconds = time.perf_counter() - photon_started

        observer_started = time.perf_counter()
        observer_shape = observer_size(target.shape)
        deterministic_labels_observer = segment_sensor_batch(
            model,
            deterministic,
            observer_shape,
            device=device,
            maximum_batch_size=maximum_batch_size,
            transform=transform,
        )
        poisson_labels_observer = segment_sensor_batch(
            model,
            poisson,
            observer_shape,
            device=device,
            maximum_batch_size=maximum_batch_size,
            transform=transform,
        )
        _synchronize(device)
        observer_seconds = time.perf_counter() - observer_started

        metrics_started = time.perf_counter()
        deterministic_labels = resample_labels(
            deterministic_labels_observer, target.shape
        )
        poisson_labels = resample_labels(poisson_labels_observer, target.shape)
        all_labels = np.concatenate([deterministic_labels, poisson_labels])
        metrics = []
        for prediction in all_labels:
            valid_target, valid_prediction = valid_region_labels(
                target, prediction, margin=48
            )
            metrics.append(instance_metrics(valid_target, valid_prediction))
        metrics_seconds = time.perf_counter() - metrics_started
        measurements.append(
            {
                "source_image_id": record.image_id,
                "source_shape": list(object_image.shape),
                "observer_shape": list(observer_shape),
                "metric_shape": [target.shape[0] - 96, target.shape[1] - 96],
                "deterministic_conditions": len(deterministic_labels),
                "poisson_conditions": len(poisson_conditions),
                "load_seconds": load_seconds,
                "optics_seconds": optics_seconds,
                "photon_seconds": photon_seconds,
                "observer_seconds": observer_seconds,
                "metrics_seconds": metrics_seconds,
                "total_route_seconds": time.perf_counter() - total_started,
                "finite_metric_count": int(
                    sum(np.isfinite(metric.panoptic_quality) for metric in metrics)
                ),
            }
        )

    route_times = [measurement["total_route_seconds"] for measurement in measurements]
    conservative_seconds = max(route_times)
    counts = evaluation_condition_counts(design_count=design_count)
    projection = project_test_runtime(
        conservative_seconds,
        available_seconds=available_seconds,
        counts=counts,
    )
    complete = all(
        measurement["deterministic_conditions"] == counts.deterministic_per_source_design
        and measurement["poisson_conditions"] == counts.poisson_per_source_design
        and measurement["finite_metric_count"]
        == counts.deterministic_per_source_design + counts.poisson_per_source_design
        for measurement in measurements
    )
    return {
        "gate": "gate6_validation_evaluation_route",
        "host": platform.platform(),
        "device": str(device),
        "model_load_seconds": model_load_seconds,
        "maximum_observer_batch_size": maximum_batch_size,
        "observer_transform": asdict(transform),
        "benchmark_sources": len(records),
        "condition_counts": counts.to_dict(),
        "measurements": measurements,
        "route_seconds": {
            "median": statistics.median(route_times),
            "maximum_used_for_projection": conservative_seconds,
        },
        "projection": projection.to_dict(),
        "passed": bool(complete and projection.passed),
    }
