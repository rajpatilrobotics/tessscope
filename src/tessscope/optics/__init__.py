"""Differentiable JAX/Chromatix microscope optics."""

from tessscope.optics.model import (
    OpticsConfig,
    cubic_phase_basis,
    cubic_support_energy_fraction,
    phase_coefficients,
    psf_sensor,
    psf_sensor_cubic,
    simulate_cubic_sensor,
    simulate_sensor,
    zernike_basis,
)
from tessscope.optics.photon import (
    expected_counts,
    keyed_poisson_seed,
    normalized_counts,
    sample_poisson_rate,
)

__all__ = [
    "OpticsConfig",
    "cubic_phase_basis",
    "cubic_support_energy_fraction",
    "phase_coefficients",
    "psf_sensor",
    "psf_sensor_cubic",
    "simulate_cubic_sensor",
    "simulate_sensor",
    "zernike_basis",
    "expected_counts",
    "keyed_poisson_seed",
    "normalized_counts",
    "sample_poisson_rate",
]
