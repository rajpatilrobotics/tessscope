"""Served NumPy/SciPy autofocus with a transparent analytic/implicit VJP."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32

from tessscope.v2.autofocus.features import SpectralFeatureConfig
from tessscope.v2.autofocus.model import AutofocusResult, autofocus_forward, autofocus_vjp


class InputSchema(BaseModel):
    support_sensor: Differentiable[Array[(None, None, None), Float32]]
    support_depth_um: Array[(None,), Float32]
    query_sensor: Differentiable[Array[(None, None, None), Float32]]
    query_depth_um: Array[(None,), Float32]
    ridge_lambda: float = Field(default=1.0, gt=0)
    radial_bins: int = Field(default=10, gt=0)
    angular_bins: int = Field(default=12, gt=0)
    log_epsilon: float = Field(default=1e-6, gt=0)
    content_epsilon: float = Field(default=1e-8, gt=0)
    global_offset: float = 0.0
    global_scale: float = Field(default=1.0, gt=0)
    stage_bound_um: float = Field(default=6.0, gt=0)


class OutputSchema(BaseModel):
    predicted_depth_um: Differentiable[Array[(None,), Float32]]
    stage_action_um: Differentiable[Array[(None,), Float32]]
    focus_mse: Differentiable[Array[(), Float32]]


DIFFERENTIABLE_INPUTS = {"support_sensor", "query_sensor"}
DIFFERENTIABLE_OUTPUTS = {"predicted_depth_um", "stage_action_um", "focus_mse"}


def _feature_config(inputs: InputSchema) -> SpectralFeatureConfig:
    return SpectralFeatureConfig(
        radial_bins=inputs.radial_bins,
        angular_bins=inputs.angular_bins,
        log_epsilon=inputs.log_epsilon,
        content_epsilon=inputs.content_epsilon,
        global_offset=inputs.global_offset,
        global_scale=inputs.global_scale,
    )


def evaluate(inputs: InputSchema) -> AutofocusResult:
    return autofocus_forward(
        np.asarray(inputs.support_sensor, dtype=np.float64),
        np.asarray(inputs.support_depth_um, dtype=np.float64),
        np.asarray(inputs.query_sensor, dtype=np.float64),
        np.asarray(inputs.query_depth_um, dtype=np.float64),
        ridge_lambda=inputs.ridge_lambda,
        feature_config=_feature_config(inputs),
        stage_bound_um=inputs.stage_bound_um,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    result = evaluate(inputs)
    return OutputSchema(
        predicted_depth_um=result.predicted_depth_um.astype(np.float32),
        stage_action_um=result.stage_action_um.astype(np.float32),
        focus_mse=np.asarray(result.focus_mse, dtype=np.float32),
    )


def _selected_vjp(
    result: AutofocusResult,
    selected_inputs: set[str],
    selected_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    if not selected_inputs.issubset(DIFFERENTIABLE_INPUTS):
        raise ValueError(f"Unknown autofocus differentiable inputs: {selected_inputs}")
    if not selected_outputs.issubset(DIFFERENTIABLE_OUTPUTS):
        raise ValueError(f"Unknown autofocus differentiable outputs: {selected_outputs}")
    predicted_cotangent = (
        np.asarray(cotangent_vector["predicted_depth_um"], dtype=np.float64)
        if "predicted_depth_um" in selected_outputs
        else None
    )
    stage_cotangent = (
        np.asarray(cotangent_vector["stage_action_um"], dtype=np.float64)
        if "stage_action_um" in selected_outputs
        else None
    )
    focus_cotangent = (
        float(np.asarray(cotangent_vector["focus_mse"]))
        if "focus_mse" in selected_outputs
        else 0.0
    )
    support_gradient, query_gradient = autofocus_vjp(
        result,
        focus_mse_cotangent=focus_cotangent,
        predicted_depth_cotangent=predicted_cotangent,
        stage_action_cotangent=stage_cotangent,
    )
    gradients = {
        "support_sensor": support_gradient.astype(np.float32),
        "query_sensor": query_gradient.astype(np.float32),
    }
    return {key: gradients[key] for key in selected_inputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return _selected_vjp(evaluate(inputs), vjp_inputs, vjp_outputs, cotangent_vector)


def _directional_outputs(
    inputs: InputSchema,
    selected_inputs: set[str],
    selected_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, np.ndarray]:
    result = evaluate(inputs)
    query_count = len(result.predicted_depth_um)
    prediction_directional = np.zeros(query_count, dtype=np.float64)
    for index in range(query_count):
        basis = np.zeros(query_count, dtype=np.float64)
        basis[index] = 1.0
        gradients = _selected_vjp(
            result,
            selected_inputs,
            {"predicted_depth_um"},
            {"predicted_depth_um": basis},
        )
        prediction_directional[index] = sum(
            float(np.sum(gradients[key] * np.asarray(tangent_vector[key])))
            for key in selected_inputs
        )
    output: dict[str, np.ndarray] = {}
    if "predicted_depth_um" in selected_outputs:
        output["predicted_depth_um"] = prediction_directional.astype(np.float32)
    if "stage_action_um" in selected_outputs:
        inside = np.abs(result.predicted_depth_um) < result.stage_bound_um
        output["stage_action_um"] = (-prediction_directional * inside).astype(np.float32)
    if "focus_mse" in selected_outputs:
        error = result.predicted_depth_um - result.query_depth_um
        directional = float(np.mean(2.0 * error * prediction_directional))
        output["focus_mse"] = np.asarray(directional, dtype=np.float32)
    return output


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return _directional_outputs(inputs, jvp_inputs, jvp_outputs, tangent_vector)


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    result = evaluate(inputs)
    output: dict[str, dict[str, np.ndarray]] = {}
    output_sizes = {
        "predicted_depth_um": len(result.predicted_depth_um),
        "stage_action_um": len(result.stage_action_um),
        "focus_mse": 1,
    }
    for output_name in jac_outputs:
        rows: dict[str, list[np.ndarray]] = {key: [] for key in jac_inputs}
        for index in range(output_sizes[output_name]):
            if output_name == "focus_mse":
                cotangent: Any = np.asarray(1.0, dtype=np.float32)
            else:
                cotangent = np.zeros(output_sizes[output_name], dtype=np.float32)
                cotangent[index] = 1.0
            gradients = _selected_vjp(
                result,
                jac_inputs,
                {output_name},
                {output_name: cotangent},
            )
            for input_name in jac_inputs:
                rows[input_name].append(gradients[input_name])
        output[output_name] = {
            input_name: (
                values[0]
                if output_name == "focus_mse"
                else np.stack(values, axis=0)
            )
            for input_name, values in rows.items()
        }
    return output


def abstract_eval(abstract_inputs):
    query_sensor = getattr(abstract_inputs, "query_sensor", None)
    if query_sensor is None:
        query_sensor = abstract_inputs["query_sensor"]
    shape = getattr(query_sensor, "shape", None)
    if shape is None:
        shape = query_sensor["shape"]
    query_count = shape[0]
    return {
        "predicted_depth_um": {"shape": [query_count], "dtype": "float32"},
        "stage_action_um": {"shape": [query_count], "dtype": "float32"},
        "focus_mse": {"shape": [], "dtype": "float32"},
    }
