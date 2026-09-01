"""Locally served JAX/Chromatix TessScope optics Tesseract."""

from typing import Any

import equinox as eqx
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

from tessscope.optics.model import phase_coefficients, simulate_sensor


class InputSchema(BaseModel):
    phase_parameters: Differentiable[Array[(6,), Float32]] = Field(
        description="Unconstrained parameters mapped into the 2.5-radian RMS ball"
    )
    object_batch: Array[(None, 256, 256), Float32] = Field(
        description="Static normalized BBBC039 biological intensity patches"
    )
    depths_um: Array[(None,), Float32] = Field(
        description="Static defocus depths in microns"
    )
    expected_photons: float = Field(default=100.0, gt=0)


class OutputSchema(BaseModel):
    sensor: Differentiable[Array[(None, None, 256, 256), Float32]]
    phase_coefficients: Differentiable[Array[(6,), Float32]]


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    return {
        "sensor": simulate_sensor(
            inputs["phase_parameters"],
            inputs["object_batch"],
            inputs["depths_um"],
            inputs["expected_photons"],
        ),
        "phase_coefficients": phase_coefficients(inputs["phase_parameters"]),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
