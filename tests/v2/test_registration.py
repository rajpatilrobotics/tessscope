"""Translation-registration checks for synthetic and invalid inputs."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from tessscope.v2.data.registration import RegistrationConfig, register_translation


def test_registration_recovers_known_shift_and_improves_correlation() -> None:
    generator = np.random.default_rng(31)
    image = ndimage.gaussian_filter(generator.normal(size=(96, 112)), sigma=2.0)
    imposed_shift = np.asarray([2.4, -3.1])
    moving = ndimage.shift(image, imposed_shift, order=3, mode="reflect")

    result = register_translation(
        image,
        moving,
        RegistrationConfig(downsample=1, upsample_factor=50),
    )

    assert np.allclose(result.shift_yx_px, -imposed_shift, atol=0.2)
    assert np.linalg.norm(result.residual_shift_yx_px) < 0.25
    assert result.correlation_after > result.correlation_before


def test_registration_identity_is_zero() -> None:
    generator = np.random.default_rng(37)
    image = generator.normal(size=(64, 80))

    result = register_translation(
        image,
        image.copy(),
        RegistrationConfig(downsample=1, upsample_factor=20),
    )

    assert result.shift_yx_px == (0.0, 0.0)
    assert result.residual_shift_yx_px == (0.0, 0.0)
    assert result.correlation_after == pytest.approx(1.0)


def test_registration_rejects_constant_images() -> None:
    image = np.ones((32, 32))
    with pytest.raises(ValueError, match="constant"):
        register_translation(image, image)
