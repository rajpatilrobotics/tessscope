"""Pure selection tests for the v2.6 sequential-mask soft gates."""

from tessscope.v2_6.selection import select_soft_rows, soft_eligible


def physical(valid: bool = True) -> dict:
    support = 0.996 if valid else 0.99
    return {
        key: {
            "physical_coefficient_norm_radians": 2.0,
            "minimum_support_energy_fraction": support,
        }
        for key in ("sensing", "capture")
    }


def metrics(final: float = 0.8) -> dict:
    return {
        "first_segmentation_loss": 1.005,
        "final_segmentation_loss": final,
        "focus_mse": 1.0,
        "residual_mae_um": 0.9,
        "normalized_residual_depth_squared": 0.05,
        "normalized_stage_action_squared": 0.2,
        "first_photon_mean": 10.0,
        "second_photon_mean": 10.0,
    }


def test_soft_eligibility_requires_every_registered_gate() -> None:
    assert soft_eligible(
        metrics(), physical(), first_limit=1.008, residual_limit=0.055
    )
    assert not soft_eligible(
        metrics(), physical(False), first_limit=1.008, residual_limit=0.055
    )
    assert not soft_eligible(
        metrics(),
        physical(),
        first_limit=1.008,
        residual_limit=0.055,
        baseline_final_limit=0.79,
    )


def test_soft_selection_uses_final_then_residual_first_and_name() -> None:
    rows = [
        {"name": "b", "eligible": True, "metrics": metrics(0.7)},
        {"name": "a", "eligible": True, "metrics": metrics(0.7)},
        {"name": "ineligible", "eligible": False, "metrics": metrics(0.6)},
    ]
    assert select_soft_rows(rows, maximum=2) == ["a", "b"]
