"""Check the served SciPy autofocus VJP against directional differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from tessscope.validation.derivatives import directional_derivative_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8403")
    parser.add_argument(
        "--output",
        default="artifacts/runs/v2/gate2/autofocus-served-derivative.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = np.random.default_rng(23)
    support = generator.normal(size=(14, 18, 20)).astype(np.float32)
    query = generator.normal(size=(5, 18, 20)).astype(np.float32)
    support_depth = np.tile(
        np.asarray([-6, -4, -2, 0, 2, 4, 6], dtype=np.float32), 2
    )
    query_depth = np.asarray([-6, -2, 0, 2, 6], dtype=np.float32)
    autofocus = Tesseract.from_url(args.url, timeout=120)
    autofocus.health()

    static_inputs = {
        "support_sensor": support,
        "support_depth_um": support_depth,
        "query_depth_um": query_depth,
        "ridge_lambda": 0.5,
        "radial_bins": 3,
        "angular_bins": 4,
    }

    def objective(query_sensor: jax.Array) -> jax.Array:
        return apply_tesseract(
            autofocus,
            {"query_sensor": query_sensor, **static_inputs},
        )["focus_mse"]

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(query))
    point = query.reshape(-1)
    report = directional_derivative_report(
        lambda candidate: float(
            objective(jnp.asarray(candidate.reshape(query.shape), dtype=jnp.float32))
        ),
        point,
        np.asarray(gradient).reshape(-1),
        seed=23,
    )
    report.update(
        {
            "component": "served_numpy_scipy_spectral_autofocus",
            "objective": "focus_mse",
            "objective_value": float(value),
            "differentiated_input": "query_sensor",
            "query_shape": list(query.shape),
            "support_shape": list(support.shape),
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "overall_median_relative_error": report[
                    "overall_median_relative_error"
                ],
                "overall_cosine_agreement": report["overall_cosine_agreement"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit("Served autofocus derivative gate failed")


if __name__ == "__main__":
    main()
