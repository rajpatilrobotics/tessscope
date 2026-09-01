"""Composed spectral autofocus forward pass and exact input-image VJP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessscope.v2.autofocus.features import (
    DEFAULT_FEATURE_CONFIG,
    SpectralFeatureConfig,
    SpectralFeatureTape,
    spectral_features,
    spectral_features_vjp,
)
from tessscope.v2.autofocus.ridge import RidgeResult, ridge_predict, ridge_predict_vjp


@dataclass(frozen=True)
class AutofocusResult:
    """Forward output plus the explicit tapes required by the analytic VJP."""

    predicted_depth_um: np.ndarray
    stage_action_um: np.ndarray
    focus_mse: float
    support_feature_tape: SpectralFeatureTape
    query_feature_tape: SpectralFeatureTape
    ridge_result: RidgeResult
    query_depth_um: np.ndarray
    stage_bound_um: float


def autofocus_forward(
    support_sensor: np.ndarray,
    support_depth_um: np.ndarray,
    query_sensor: np.ndarray,
    query_depth_um: np.ndarray,
    *,
    ridge_lambda: float,
    feature_config: SpectralFeatureConfig = DEFAULT_FEATURE_CONFIG,
    stage_bound_um: float = 6.0,
) -> AutofocusResult:
    """Refit signed-z ridge weights and evaluate query focus/stage outputs."""
    support_depth = np.asarray(support_depth_um, dtype=np.float64)
    query_depth = np.asarray(query_depth_um, dtype=np.float64)
    if support_depth.ndim != 1 or query_depth.ndim != 1:
        raise ValueError("Autofocus depths must be one-dimensional")
    if len(support_sensor) != len(support_depth) or len(query_sensor) != len(query_depth):
        raise ValueError("Autofocus image and depth batch sizes differ")
    if stage_bound_um <= 0:
        raise ValueError("Stage bound must be positive")
    support_features, support_tape = spectral_features(support_sensor, feature_config)
    query_features, query_tape = spectral_features(query_sensor, feature_config)
    ridge = ridge_predict(support_features, support_depth, query_features, ridge_lambda)
    error = ridge.predictions - query_depth
    return AutofocusResult(
        predicted_depth_um=ridge.predictions,
        stage_action_um=np.clip(-ridge.predictions, -stage_bound_um, stage_bound_um),
        focus_mse=float(np.mean(error * error)),
        support_feature_tape=support_tape,
        query_feature_tape=query_tape,
        ridge_result=ridge,
        query_depth_um=query_depth,
        stage_bound_um=stage_bound_um,
    )


def autofocus_vjp(
    result: AutofocusResult,
    *,
    focus_mse_cotangent: float = 0.0,
    predicted_depth_cotangent: np.ndarray | None = None,
    stage_action_cotangent: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact support/query sensor cotangents for selected outputs."""
    prediction_cotangent = np.zeros_like(result.predicted_depth_um)
    if predicted_depth_cotangent is not None:
        prediction_cotangent += np.asarray(predicted_depth_cotangent, dtype=np.float64)
    error = result.predicted_depth_um - result.query_depth_um
    prediction_cotangent += float(focus_mse_cotangent) * 2.0 * error / len(error)
    if stage_action_cotangent is not None:
        stage_cotangent = np.asarray(stage_action_cotangent, dtype=np.float64)
        if stage_cotangent.shape != result.stage_action_um.shape:
            raise ValueError("Stage-action cotangent shape differs from stage output")
        inside = np.abs(result.predicted_depth_um) < result.stage_bound_um
        prediction_cotangent -= stage_cotangent * inside
    support_feature_cotangent, query_feature_cotangent = ridge_predict_vjp(
        result.ridge_result,
        prediction_cotangent,
    )
    return (
        spectral_features_vjp(result.support_feature_tape, support_feature_cotangent),
        spectral_features_vjp(result.query_feature_tape, query_feature_cotangent),
    )
