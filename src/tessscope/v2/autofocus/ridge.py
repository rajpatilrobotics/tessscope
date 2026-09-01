"""Regularized signed-z ridge fit with a hand-derived implicit adjoint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve


@dataclass(frozen=True)
class RidgeResult:
    """Forward values needed by the exact implicit ridge VJP."""

    support_augmented: np.ndarray
    query_augmented: np.ndarray
    targets: np.ndarray
    gram: np.ndarray
    weights: np.ndarray
    predictions: np.ndarray
    residual: np.ndarray


def ridge_predict(
    support_features: np.ndarray,
    support_targets: np.ndarray,
    query_features: np.ndarray,
    ridge_lambda: float,
) -> RidgeResult:
    """Fit an unregularized intercept and regularized feature weights."""
    support = np.asarray(support_features, dtype=np.float64)
    targets = np.asarray(support_targets, dtype=np.float64)
    query = np.asarray(query_features, dtype=np.float64)
    if support.ndim != 2 or query.ndim != 2 or targets.ndim != 1:
        raise ValueError("Ridge inputs must have shapes S,F; S; Q,F")
    if support.shape[0] != targets.shape[0] or support.shape[1] != query.shape[1]:
        raise ValueError("Ridge support, target, and query dimensions are inconsistent")
    if ridge_lambda <= 0:
        raise ValueError("Ridge lambda must be positive")
    support_augmented = np.concatenate(
        [support, np.ones((len(support), 1), dtype=np.float64)], axis=1
    )
    query_augmented = np.concatenate(
        [query, np.ones((len(query), 1), dtype=np.float64)], axis=1
    )
    penalty = np.eye(support_augmented.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    gram = support_augmented.T @ support_augmented + ridge_lambda * penalty
    weights = solve(
        gram,
        support_augmented.T @ targets,
        assume_a="sym",
        check_finite=False,
    )
    predictions = query_augmented @ weights
    residual = targets - support_augmented @ weights
    return RidgeResult(
        support_augmented=support_augmented,
        query_augmented=query_augmented,
        targets=targets,
        gram=gram,
        weights=weights,
        predictions=predictions,
        residual=residual,
    )


def ridge_predict_vjp(
    result: RidgeResult,
    prediction_cotangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return support/query feature cotangents through the implicit ridge solve."""
    cotangent = np.asarray(prediction_cotangent, dtype=np.float64)
    if cotangent.shape != result.predictions.shape:
        raise ValueError(
            f"Expected prediction cotangent {result.predictions.shape}, got {cotangent.shape}"
        )
    weight_cotangent = result.query_augmented.T @ cotangent
    implicit = solve(
        result.gram.T,
        weight_cotangent,
        assume_a="sym",
        check_finite=False,
    )
    query_augmented_cotangent = np.outer(cotangent, result.weights)
    support_augmented_cotangent = np.outer(result.residual, implicit) - np.outer(
        result.support_augmented @ implicit,
        result.weights,
    )
    # The final augmented column is the fixed intercept input and is not differentiable.
    return support_augmented_cotangent[:, :-1], query_augmented_cotangent[:, :-1]
