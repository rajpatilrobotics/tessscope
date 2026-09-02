"""Pure projection and selection-adjacent tests for two-mask optimization."""

import numpy as np
import pytest

from tessscope.v2_1.optics.model import phase_coefficients_b7
from tessscope.v2_6.optimization import feasibility_violations, project_two_masks


def test_two_mask_projection_applies_independent_b7_bounds() -> None:
    parameters = np.concatenate(
        [np.full(7, 30.0, dtype=np.float32), np.full(7, -40.0, dtype=np.float32)]
    )
    projected, changed = project_two_masks(parameters)
    assert changed == (True, True)
    assert np.linalg.norm(np.asarray(phase_coefficients_b7(projected[:7]))) == pytest.approx(
        2.4975, abs=1e-5
    )
    assert np.linalg.norm(np.asarray(phase_coefficients_b7(projected[7:]))) == pytest.approx(
        2.4975, abs=1e-5
    )


def test_feasibility_violations_keep_registered_scales() -> None:
    observed = feasibility_violations(
        {
            "first_segmentation_loss": 1.104,
            "normalized_residual_depth_squared": 0.065,
        },
        first_limit=1.10,
        residual_limit=0.055,
    )
    assert observed == pytest.approx([0.5, 0.5])
