"""Selection tests for the frozen v2.5 soft rules."""

from tessscope.v2_5.selection import is_soft_eligible, select_candidates


def row(name: str, start: str, first: float, final: float, residual: float) -> dict:
    value = {
        "name": name,
        "start_name": start,
        "parameter_sha256": name,
        "basis": "B11",
        "finite_parameters": True,
        "first_segmentation_loss": first,
        "final_segmentation_loss": final,
        "residual_mae_um": residual,
        "normalized_residual_depth_squared": 0.05,
        "start_residual_mae_um": 1.5,
        "residual_bound": 0.055,
        "physical_coefficient_norm_radians": 1.0,
        "minimum_support_energy_fraction": 0.997,
    }
    value["eligible"] = is_soft_eligible(
        value, first_segmentation_maximum=1.0981961773633957
    )
    return value


def test_soft_gate_requires_basis_constraints_improvement_and_physics() -> None:
    candidate = row("candidate", "a", 1.095, 1.04, 1.1)
    assert candidate["eligible"]
    candidate["basis"] = "B7"
    assert not is_soft_eligible(
        candidate, first_segmentation_maximum=1.0981961773633957
    )
    candidate["basis"] = "B11"
    candidate["normalized_residual_depth_squared"] = 0.06
    assert not is_soft_eligible(
        candidate, first_segmentation_maximum=1.0981961773633957
    )


def test_selection_enforces_nondominance_and_start_diversity() -> None:
    rows = [
        row("a-best", "a", 1.09, 1.02, 1.2),
        row("a-other", "a", 1.08, 1.04, 1.0),
        row("b", "b", 1.095, 1.01, 1.1),
        row("dominated", "c", 1.097, 1.05, 1.3),
    ]
    selection = select_candidates(rows)
    assert "dominated" not in selection["nondominated_names"]
    assert len(selection["selected_names"]) == 2
    assert "b" in selection["selected_names"]
