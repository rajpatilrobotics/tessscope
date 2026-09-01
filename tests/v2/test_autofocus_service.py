"""Direct API parity and VJP check for the SciPy autofocus Tesseract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "services"
    / "v2"
    / "autofocus"
    / "tesseract_api.py"
)
SPEC = importlib.util.spec_from_file_location("tessscope_v2_autofocus_service", SERVICE_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVICE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVICE
SPEC.loader.exec_module(SERVICE)


def test_autofocus_service_vjp_matches_float32_directional_difference() -> None:
    generator = np.random.default_rng(23)
    support = generator.normal(size=(14, 18, 20)).astype(np.float32)
    query = generator.normal(size=(5, 18, 20)).astype(np.float32)
    support_depth = np.tile(np.asarray([-6, -4, -2, 0, 2, 4, 6]), 2).astype(
        np.float32
    )
    query_depth = np.asarray([-6, -2, 0, 2, 6], dtype=np.float32)
    direction = generator.choice((-1.0, 1.0), size=query.shape).astype(np.float32)

    def inputs(value: np.ndarray):
        return SERVICE.InputSchema(
            support_sensor=support,
            support_depth_um=support_depth,
            query_sensor=value,
            query_depth_um=query_depth,
            ridge_lambda=0.5,
            radial_bins=3,
            angular_bins=4,
        )

    base = inputs(query)
    gradient = SERVICE.vector_jacobian_product(
        base,
        {"query_sensor"},
        {"focus_mse"},
        {"focus_mse": np.asarray(1.0, dtype=np.float32)},
    )["query_sensor"]
    analytic = float(np.sum(gradient * direction))
    epsilon = 1e-3
    finite = float(
        (
            np.asarray(SERVICE.apply(inputs(query + epsilon * direction)).focus_mse)
            - np.asarray(SERVICE.apply(inputs(query - epsilon * direction)).focus_mse)
        )
        / (2 * epsilon)
    )
    error = abs(analytic - finite) / max(abs(analytic), abs(finite), 1e-8)
    assert error < 2e-3


def test_autofocus_service_abstract_eval_accepts_runtime_objects() -> None:
    abstract_inputs = SimpleNamespace(
        query_sensor=SimpleNamespace(shape=(5, 18, 20), dtype="float32")
    )

    outputs = SERVICE.abstract_eval(abstract_inputs)

    assert outputs["predicted_depth_um"]["shape"] == [5]
    assert outputs["stage_action_um"]["shape"] == [5]
    assert outputs["focus_mse"]["shape"] == []
