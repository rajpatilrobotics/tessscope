"""PyTorch/InstanSeg observer schema for the seven-coefficient v2.1 pupil."""

from __future__ import annotations

from functools import cache
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Int32, UInt8

from tessscope.observer.instanseg import load_frozen_model
from tessscope.observer.loss import (
    PHASE_L2_WEIGHT,
    ObserverTransform,
    design_task_loss,
    surrogate_proxy_loss,
)


class InputSchema(BaseModel):
    sensor: Differentiable[Array[(None, None, 256, 256), Float32]]
    phase_coefficients: Differentiable[Array[(7,), Float32]]
    instance_labels: Array[(None, 330, 330), Int32]
    centers_yx: Array[(None, 32, 2), Float32]
    valid_objects: Array[(None, 32), UInt8]
    transform_mode: Literal["identity", "affine", "asinh"] = "identity"
    transform_offset: float = 0.0
    transform_scale: float = Field(default=1.0, gt=0)
    backward_mode: Literal["exact", "surrogate"] = "exact"
    surrogate_scale: float = Field(default=1.0, gt=0)


class OutputSchema(BaseModel):
    task_loss: Differentiable[Array[(), Float32]]


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@cache
def _model(device_name: str) -> torch.nn.Module:
    return load_frozen_model(device_name)


def _to_tensor(value: Any, device: torch.device) -> Any:
    if isinstance(value, np.ndarray | np.generic):
        array = np.asarray(value)
        if not array.flags.writeable:
            array = array.copy()
        return torch.as_tensor(array, device=device)
    return value


def _tensor_inputs(
    inputs: InputSchema,
    differentiable: set[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    device = _device()
    payload = {key: _to_tensor(value, device) for key, value in inputs.model_dump().items()}
    for key in differentiable:
        value = payload[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Differentiable input {key} is not an array")
        payload[key] = value.detach().requires_grad_(True)
    return payload


def evaluate(inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    transform = ObserverTransform(
        mode=inputs["transform_mode"],
        offset=inputs["transform_offset"],
        scale=inputs["transform_scale"],
    )
    device = inputs["sensor"].device
    loss = design_task_loss(
        _model(str(device)),
        inputs["sensor"],
        inputs["phase_coefficients"],
        inputs["instance_labels"].to(torch.long),
        inputs["centers_yx"],
        inputs["valid_objects"].to(torch.bool),
        transform,
    )
    return {"task_loss": loss}


def _loss_and_gradients(
    inputs: InputSchema,
    selected_inputs: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    differentiable_inputs = set(selected_inputs)
    if inputs.backward_mode == "surrogate":
        differentiable_inputs.add("sensor")
    payload = _tensor_inputs(inputs, differentiable_inputs)
    loss = evaluate(payload)["task_loss"]
    selected_tensors = tuple(payload[key] for key in selected_inputs)
    if payload["backward_mode"] == "exact":
        gradients = torch.autograd.grad(loss, selected_tensors, allow_unused=True)
    else:
        transform = ObserverTransform(
            mode=payload["transform_mode"],
            offset=payload["transform_offset"],
            scale=payload["transform_scale"],
        )
        proxy = surrogate_proxy_loss(
            payload["sensor"],
            payload["instance_labels"].to(torch.long),
            transform,
        )
        sensor_gradient = (
            torch.autograd.grad(
                proxy,
                payload["sensor"],
                retain_graph=True,
            )[0]
            * payload["surrogate_scale"]
        )
        gradients = tuple(
            sensor_gradient
            if key == "sensor"
            else (
                2.0 * PHASE_L2_WEIGHT * payload["phase_coefficients"]
                if key == "phase_coefficients"
                else None
            )
            for key in selected_inputs
        )
    return loss, {
        key: torch.zeros_like(value) if gradient is None else gradient
        for key, value, gradient in zip(
            selected_inputs,
            selected_tensors,
            gradients,
            strict=True,
        )
    }


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().to("cpu").numpy()


def apply(inputs: InputSchema) -> OutputSchema:
    with torch.no_grad():
        output = evaluate(_tensor_inputs(inputs))
    return OutputSchema(task_loss=_numpy(output["task_loss"]))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    if jac_outputs != {"task_loss"}:
        raise ValueError("The observer has only the differentiable task_loss output")
    ordered_inputs = tuple(sorted(jac_inputs))
    _, gradients = _loss_and_gradients(inputs, ordered_inputs)
    return {"task_loss": {key: _numpy(gradients[key]) for key in ordered_inputs}}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    if jvp_outputs != {"task_loss"}:
        raise ValueError("The observer has only the differentiable task_loss output")
    ordered_inputs = tuple(sorted(jvp_inputs))
    _, gradients = _loss_and_gradients(inputs, ordered_inputs)
    directional = sum(
        (gradients[key] * torch.as_tensor(tangent_vector[key], device=gradients[key].device)).sum()
        for key in ordered_inputs
    )
    return {"task_loss": _numpy(directional)}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    if vjp_outputs != {"task_loss"}:
        raise ValueError("The observer has only the differentiable task_loss output")
    ordered_inputs = tuple(sorted(vjp_inputs))
    _, gradients = _loss_and_gradients(inputs, ordered_inputs)
    cotangent = float(np.asarray(cotangent_vector["task_loss"]))
    return {key: _numpy(cotangent * gradients[key]) for key in ordered_inputs}


def abstract_eval(abstract_inputs):
    return {"task_loss": {"shape": [], "dtype": "float32"}}
