"""Minimal deterministic Adam update used by every learned design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdamState:
    step: int
    first_moment: np.ndarray
    second_moment: np.ndarray

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> AdamState:
        return cls(
            step=0,
            first_moment=np.zeros(shape, dtype=np.float32),
            second_moment=np.zeros(shape, dtype=np.float32),
        )


def adam_update(
    parameters: np.ndarray,
    gradient: np.ndarray,
    state: AdamState,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, AdamState]:
    """Apply one conventional Adam step with bias correction."""
    if learning_rate <= 0:
        raise ValueError("Adam learning rate must be positive")
    if not 0 < beta1 < 1 or not 0 < beta2 < 1:
        raise ValueError("Adam beta values must lie in (0, 1)")
    value = np.asarray(parameters, dtype=np.float32)
    grad = np.asarray(gradient, dtype=np.float32)
    if value.shape != grad.shape or value.shape != state.first_moment.shape:
        raise ValueError("Adam parameter, gradient, and state shapes must match")
    step = state.step + 1
    first = beta1 * state.first_moment + (1.0 - beta1) * grad
    second = beta2 * state.second_moment + (1.0 - beta2) * np.square(grad)
    corrected_first = first / (1.0 - beta1**step)
    corrected_second = second / (1.0 - beta2**step)
    updated = value - learning_rate * corrected_first / (
        np.sqrt(corrected_second) + epsilon
    )
    return updated.astype(np.float32), AdamState(step, first, second)
