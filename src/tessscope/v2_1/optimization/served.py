"""Served B7 branch averaging shared by constrained experiments."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tessscope.v2.optimization.objective import JointObjectiveWeights
from tessscope.v2.optimization.served import (
    V2SystemCalibration,
    joint_value,
    joint_value_and_gradient,
)
from tessscope.v2_1.optics.model import phase_coefficients_b7
from tessscope.v2_1.optimization.constrained import BranchEvaluation


def b7_coefficient_list(parameters: np.ndarray) -> list[float]:
    """Return physical B7 coefficients for one unconstrained parameter vector."""
    return np.asarray(phase_coefficients_b7(jnp.asarray(parameters))).tolist()


def averaged_branches(
    services,
    parameters: np.ndarray,
    batches,
    calibration: V2SystemCalibration,
) -> BranchEvaluation:
    """Average exact segmentation and focus values/Jacobians over served batches."""
    segmentation_values = []
    focus_values = []
    segmentation_gradients = []
    focus_gradients = []
    for batch in batches:
        _, segmentation_gradient, segmentation = joint_value_and_gradient(
            *services,
            np.asarray(parameters, dtype=np.float32),
            batch,
            calibration,
            JointObjectiveWeights(segmentation=1.0, focus=0.0),
        )
        _, focus_gradient, focus = joint_value_and_gradient(
            *services,
            np.asarray(parameters, dtype=np.float32),
            batch,
            calibration,
            JointObjectiveWeights(segmentation=0.0, focus=1.0),
        )
        segmentation_values.append(segmentation["segmentation_loss"])
        focus_values.append(focus["focus_mse"])
        segmentation_gradients.append(segmentation_gradient)
        focus_gradients.append(focus_gradient)
    return BranchEvaluation(
        segmentation_loss=float(np.mean(segmentation_values)),
        focus_mse=float(np.mean(focus_values)),
        segmentation_gradient=np.mean(segmentation_gradients, axis=0),
        normalized_focus_gradient=np.mean(focus_gradients, axis=0),
    )


def averaged_forward(
    services,
    parameters: np.ndarray,
    batches,
    calibration: V2SystemCalibration,
) -> dict[str, float]:
    """Average served forward branch diagnostics without requesting Jacobians."""
    rows = []
    for batch in batches:
        _, branches = joint_value(
            *services,
            np.asarray(parameters, dtype=np.float32),
            batch,
            calibration,
            JointObjectiveWeights(),
        )
        rows.append(branches)
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
