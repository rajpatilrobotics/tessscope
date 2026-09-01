"""Finite-difference tests for the SciPy spectral and implicit-ridge adjoints."""

from __future__ import annotations

import numpy as np

from tessscope.v2.autofocus.features import (
    SpectralFeatureConfig,
    spectral_features,
    spectral_features_vjp,
)
from tessscope.v2.autofocus.model import autofocus_forward, autofocus_vjp
from tessscope.v2.autofocus.ridge import ridge_predict, ridge_predict_vjp


def central_directional(function, point, direction, epsilon=1e-6) -> float:
    return float(
        (function(point + epsilon * direction) - function(point - epsilon * direction))
        / (2 * epsilon)
    )


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def test_spectral_feature_vjp_matches_central_difference() -> None:
    generator = np.random.default_rng(11)
    images = generator.normal(size=(2, 19, 17))
    direction = generator.normal(size=images.shape)
    cotangent = generator.normal(size=(2, 12))
    config = SpectralFeatureConfig(
        radial_bins=3,
        angular_bins=4,
        global_offset=0.2,
        global_scale=1.7,
    )
    features, tape = spectral_features(images, config)
    gradient = spectral_features_vjp(tape, cotangent)
    analytic = float(np.sum(gradient * direction))
    finite = central_directional(
        lambda value: np.sum(spectral_features(value, config)[0] * cotangent),
        images,
        direction,
    )
    assert features.shape == (2, 12)
    assert relative_error(analytic, finite) < 2e-6


def test_implicit_ridge_vjp_matches_support_and_query_directions() -> None:
    generator = np.random.default_rng(13)
    support = generator.normal(size=(11, 5))
    targets = generator.normal(size=(11,))
    query = generator.normal(size=(4, 5))
    support_direction = generator.normal(size=support.shape)
    query_direction = generator.normal(size=query.shape)
    cotangent = generator.normal(size=(4,))
    ridge_lambda = 0.7
    result = ridge_predict(support, targets, query, ridge_lambda)
    support_gradient, query_gradient = ridge_predict_vjp(result, cotangent)
    finite_support = central_directional(
        lambda value: np.sum(
            ridge_predict(value, targets, query, ridge_lambda).predictions * cotangent
        ),
        support,
        support_direction,
    )
    finite_query = central_directional(
        lambda value: np.sum(
            ridge_predict(support, targets, value, ridge_lambda).predictions * cotangent
        ),
        query,
        query_direction,
    )
    support_error = relative_error(
        float(np.sum(support_gradient * support_direction)), finite_support
    )
    assert support_error < 2e-7
    assert relative_error(float(np.sum(query_gradient * query_direction)), finite_query) < 2e-7


def test_full_autofocus_vjp_matches_central_difference() -> None:
    generator = np.random.default_rng(17)
    support = generator.normal(size=(14, 18, 20))
    query = generator.normal(size=(5, 18, 20))
    support_depth = np.tile(np.asarray([-6, -4, -2, 0, 2, 4, 6]), 2)
    query_depth = np.asarray([-6, -2, 0, 2, 6], dtype=np.float64)
    support_direction = generator.normal(size=support.shape)
    query_direction = generator.normal(size=query.shape)
    config = SpectralFeatureConfig(radial_bins=3, angular_bins=4)
    result = autofocus_forward(
        support,
        support_depth,
        query,
        query_depth,
        ridge_lambda=0.5,
        feature_config=config,
    )
    support_gradient, query_gradient = autofocus_vjp(
        result,
        focus_mse_cotangent=1.0,
    )
    finite_support = central_directional(
        lambda value: autofocus_forward(
            value,
            support_depth,
            query,
            query_depth,
            ridge_lambda=0.5,
            feature_config=config,
        ).focus_mse,
        support,
        support_direction,
    )
    finite_query = central_directional(
        lambda value: autofocus_forward(
            support,
            support_depth,
            value,
            query_depth,
            ridge_lambda=0.5,
            feature_config=config,
        ).focus_mse,
        query,
        query_direction,
    )
    analytic_support = float(np.sum(support_gradient * support_direction))
    analytic_query = float(np.sum(query_gradient * query_direction))
    assert relative_error(analytic_support, finite_support) < 3e-6
    assert relative_error(analytic_query, finite_query) < 3e-6
