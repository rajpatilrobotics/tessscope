"""B7 optics with differentiable depth input for the v2.3 feedback loop."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

from tessscope.v2_1.optics.model import (
    phase_coefficients_b7,
    simulate_noisy_sensor_b7,
)


class InputSchema(BaseModel):
    phase_parameters: Differentiable[Array[(7,), Float32]]
    object_batch: Array[(None, 256, 256), Float32]
    depths_um: Differentiable[Array[(None,), Float32]]
    noise_standard_normal: Array[(None, None, 256, 256), Float32]
    expected_photons: float = Field(default=200.0, gt=0)
    exposure_gain: float = Field(default=1.0, gt=0)
    depth_scale: float = Field(default=1.0, gt=0)
    axial_offset_um: float = 0.0


class OutputSchema(BaseModel):
    sensor: Differentiable[Array[(None, None, 256, 256), Float32]]
    phase_coefficients: Differentiable[Array[(7,), Float32]]
    photon_mean: Differentiable[Array[(), Float32]]


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    sensor = simulate_noisy_sensor_b7(
        inputs["phase_parameters"],
        inputs["object_batch"],
        inputs["depths_um"],
        inputs["noise_standard_normal"],
        expected_photons=inputs["expected_photons"],
        exposure_gain=inputs["exposure_gain"],
        depth_scale=inputs["depth_scale"],
        axial_offset_um=inputs["axial_offset_um"],
    )
    return {
        "sensor": sensor,
        "phase_coefficients": phase_coefficients_b7(inputs["phase_parameters"]),
        "photon_mean": jnp.mean(sensor) * inputs["expected_photons"],
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
