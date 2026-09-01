"""Directional finite-difference checks with frozen TessScope thresholds."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def directional_derivative_report(
    objective: Callable[[np.ndarray], float],
    point: np.ndarray,
    gradient: np.ndarray,
    *,
    seed: int = 29,
    direction_count: int = 5,
    epsilons: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4),
    maximum_median_relative_error: float = 1e-2,
    minimum_cosine_agreement: float = 0.99,
    minimum_stable_epsilons: int = 3,
) -> dict[str, object]:
    """Compare a supplied gradient with central differences in fixed directions."""
    point = np.asarray(point, dtype=np.float32)
    gradient = np.asarray(gradient, dtype=np.float64)
    generator = np.random.default_rng(seed)
    # Rademacher directions are conventional for simultaneous-perturbation
    # checks and avoid unstable relative errors from near-zero projections.
    directions = generator.choice((-1.0, 1.0), size=(direction_count, point.size))
    analytic = directions @ gradient.reshape(-1)

    by_epsilon: dict[str, object] = {}
    aligned_values: list[tuple[float, np.ndarray, np.ndarray]] = []
    for epsilon in epsilons:
        finite_difference = np.asarray(
            [
                (
                    objective(point + epsilon * direction)
                    - objective(point - epsilon * direction)
                )
                / (2.0 * epsilon)
                for direction in directions
            ],
            dtype=np.float64,
        )
        denominator = np.maximum.reduce(
            [
                np.abs(analytic),
                np.abs(finite_difference),
                np.full(direction_count, 1e-8),
            ]
        )
        relative_error = np.abs(finite_difference - analytic) / denominator
        norm_product = np.linalg.norm(analytic) * np.linalg.norm(finite_difference)
        cosine = (
            float(np.dot(analytic, finite_difference) / norm_product)
            if norm_product > 0
            else 1.0
        )
        median_relative_error = float(np.median(relative_error))
        aligned = bool(
            np.isfinite(finite_difference).all()
            and cosine > minimum_cosine_agreement
        )
        if aligned:
            aligned_values.append((epsilon, finite_difference, relative_error))
        by_epsilon[f"{epsilon:.0e}"] = {
            "finite_difference": finite_difference.tolist(),
            "relative_error": relative_error.tolist(),
            "median_relative_error": median_relative_error,
            "cosine_agreement": cosine,
            "cosine_aligned": aligned,
            "in_stable_window": False,
        }

    # If more than the required count align, use the smallest aligned epsilon
    # window. This avoids letting the known truncation regime at the largest
    # step dominate the median while retaining the frozen epsilon sweep.
    selected_values = aligned_values[-minimum_stable_epsilons:]
    stable_epsilons = [value[0] for value in selected_values]
    stable_finite_differences = [value[1] for value in selected_values]
    stable_relative_errors = [value[2] for value in selected_values]
    stable_analytic = [analytic for _ in selected_values]
    for epsilon in stable_epsilons:
        by_epsilon[f"{epsilon:.0e}"]["in_stable_window"] = True  # type: ignore[index]

    if stable_epsilons:
        combined_finite_difference = np.concatenate(stable_finite_differences)
        combined_analytic = np.concatenate(stable_analytic)
        combined_relative_error = np.concatenate(stable_relative_errors)
        combined_norm_product = np.linalg.norm(combined_analytic) * np.linalg.norm(
            combined_finite_difference
        )
        overall_cosine = float(
            np.dot(combined_analytic, combined_finite_difference)
            / combined_norm_product
        )
        overall_median_relative_error = float(np.median(combined_relative_error))
    else:
        overall_cosine = float("nan")
        overall_median_relative_error = float("inf")

    passed = bool(
        len(stable_epsilons) >= minimum_stable_epsilons
        and overall_median_relative_error < maximum_median_relative_error
        and overall_cosine > minimum_cosine_agreement
    )
    return {
        "seed": seed,
        "directions": directions.tolist(),
        "analytic_gradient": gradient.tolist(),
        "analytic_directional_derivatives": analytic.tolist(),
        "epsilons": by_epsilon,
        "thresholds": {
            "maximum_median_relative_error": maximum_median_relative_error,
            "minimum_cosine_agreement": minimum_cosine_agreement,
            "minimum_stable_epsilons": minimum_stable_epsilons,
        },
        "stable_epsilons": stable_epsilons,
        "overall_median_relative_error": overall_median_relative_error,
        "overall_cosine_agreement": overall_cosine,
        "preferred_relative_error_target_met": overall_median_relative_error < 1e-3,
        "passed": passed,
    }
