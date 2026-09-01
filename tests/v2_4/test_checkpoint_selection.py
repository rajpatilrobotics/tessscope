"""Pure tests for preregistered v2.4 selection and diversity."""

from tessscope.v2_4.checkpoints import (
    FIRST_SEGMENTATION_MAXIMUM,
    eligible_intervals,
    is_soft_eligible,
    select_checkpoints,
)


def row(name: str, run: str, parameter_hash: str, first: float, final: float, residual: float):
    return {
        "name": name,
        "source_run": run,
        "parameter_sha256": parameter_hash,
        "objective": 1.0,
        "first_segmentation_loss": first,
        "final_segmentation_loss": final,
        "focus_mse": 1.0,
        "residual_mae_um": residual,
        "start_residual_mae_um": 2.0,
        "normalized_residual_depth_squared": 0.1,
        "normalized_stage_action_squared": 0.1,
        "first_photon_mean": 100.0,
        "second_photon_mean": 100.0,
        "seven_finite_phase_parameters": True,
        "physical_coefficient_norm_radians": 1.0,
        "minimum_support_energy_fraction": 0.997,
    }


def test_eligibility_keeps_ceiling_unchanged_and_requires_residual_gain() -> None:
    candidate = row("candidate", "run", "hash", FIRST_SEGMENTATION_MAXIMUM, 1.0, 1.0)
    assert is_soft_eligible(candidate)
    candidate["first_segmentation_loss"] += 1e-12
    assert not is_soft_eligible(candidate)
    candidate["first_segmentation_loss"] = FIRST_SEGMENTATION_MAXIMUM
    candidate["residual_mae_um"] = candidate["start_residual_mae_um"]
    assert not is_soft_eligible(candidate)


def test_selection_deduplicates_hashes_and_allows_one_checkpoint_per_run() -> None:
    rows = [
        row("a-step-01", "a", "duplicate", 1.08, 1.02, 1.0),
        row("b-step-01", "b", "duplicate", 1.08, 1.02, 1.0),
        row("a-step-02", "a", "a2", 1.07, 1.01, 0.9),
        row("c-step-01", "c", "c1", 1.06, 1.04, 0.7),
        row("dominated", "d", "d1", 1.09, 1.2, 1.2),
    ]
    for candidate in rows:
        candidate["eligible"] = is_soft_eligible(candidate)
    selection = select_checkpoints(rows)
    assert selection["canonical_eligible_names"] == [
        "a-step-01",
        "a-step-02",
        "c-step-01",
        "dominated",
    ]
    assert selection["nondominated_names"] == ["a-step-02", "c-step-01"]
    assert selection["selected_names"] == ["a-step-02", "c-step-01"]


def test_eligible_intervals_preserve_disjoint_gate_windows() -> None:
    rows = [
        {"step": 1, "eligible": True},
        {"step": 2, "eligible": True},
        {"step": 3, "eligible": False},
        {"step": 4, "eligible": True},
        {"step": 5, "eligible": True},
    ]
    assert eligible_intervals(rows) == [[1, 2], [4, 5]]
