"""Check the 96-pixel PSF support against a 2× larger reference."""

import json

import jax.numpy as jnp

from tessscope.optics.model import psf_support_energy_fraction


def main() -> None:
    parameters = jnp.zeros((6,), dtype=jnp.float32)
    depths = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
    fractions = {
        str(depth): float(psf_support_energy_fraction(parameters, jnp.asarray(depth)))
        for depth in depths
    }
    result = {
        "support_px": 96,
        "oversampling": 4,
        "reference_factor": 2,
        "minimum_required_fraction": 0.995,
        "clear_pupil_fractions": fractions,
        "minimum_fraction": min(fractions.values()),
        "passed": min(fractions.values()) >= 0.995,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("96-pixel support failed; switch the frozen config to 128")


if __name__ == "__main__":
    main()
