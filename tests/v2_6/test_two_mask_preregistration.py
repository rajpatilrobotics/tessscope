"""Integrity checks for the v2.6 sequential two-mask preregistration."""

import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "configs" / "v2_6" / "two-mask-contract.yaml"
SOURCES = PROJECT_ROOT / "configs" / "v2_6" / "two-mask-sources.json"


def test_two_mask_contract_freezes_budget_gates_and_test_seal() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["status"] == "preregistered_before_two_mask_gradient_or_optimization"
    assert contract["physical_assumptions"]["masks"] == 2
    assert contract["physical_assumptions"]["fixed_total_expected_photons"] == 400.0
    assert contract["matched_piecewise_family"]["total_pairs"] == 76
    assert contract["feasibility_restoration"]["before_final_task_optimization"] is True
    assert contract["primary_optimizer"]["total_primary_steps_maximum"] == 72
    assert contract["hard_protocol_if_selected"][
        "minimum_corrected_pq_gain_over_best_two_mask_piecewise"
    ] == 0.005
    assert contract["test_policy"]["access_runs"] == 1
    assert contract["test_accessed"] is False


def test_two_mask_sources_freeze_four_b7_vectors() -> None:
    sources = json.loads(SOURCES.read_text())
    assert set(sources["parameters"]) == {
        "b7_segmentation_only",
        "b7_focus_only",
        "v2_2_piecewise_028",
        "v2_4_balanced_step14",
    }
    assert all(len(vector) == 7 for vector in sources["parameters"].values())
    assert sources["test_accessed"] is False
