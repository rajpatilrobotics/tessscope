"""Integrity checks for the v2.5 contract before new pupil generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "configs" / "v2_5" / "contract.yaml"
MANIFEST = PROJECT_ROOT / "configs" / "v2_5" / "source-manifest.json"


def parameter_sha256(parameters: list[float]) -> str:
    payload = json.dumps(parameters, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def test_v2_5_source_files_and_start_hashes_are_frozen() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["source_files"].items():
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == expected
    assert len(manifest["starts"]) == 7
    assert len({row["name"] for row in manifest["starts"]}) == 7
    for row in manifest["starts"]:
        assert parameter_sha256(row["parameters"]) == row["parameter_sha256"]
        assert len(row["parameters"]) == (7 if row["basis"] == "B7" else 11)


def test_v2_5_b7_to_b11_lifts_are_exact_zero_padding() -> None:
    starts = {
        row["name"]: row["parameters"]
        for row in json.loads(MANIFEST.read_text())["starts"]
    }
    assert starts["b11_lift_v2_4_balanced_step14"] == [
        *starts["b7_v2_4_balanced_step14"],
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert starts["b11_lift_v2_2_piecewise_028"] == [
        *starts["b7_v2_2_piecewise_028"],
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_v2_5_contract_freezes_budgets_gates_and_test_seal() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    assert contract["status"] == "preregistered_before_new_pupil_generation"
    assert contract["primary_optimizer"]["residual_epsilon_ladder"] == [
        0.075,
        0.065,
        0.055,
    ]
    assert contract["primary_optimizer"]["steps_per_epsilon_stage"] == 18
    assert contract["primary_optimizer"]["constraint_aggregation"] == (
        "exact_mean_of_all_four_frozen_training_batches_each_step"
    )
    assert contract["soft_selection"]["maximum_promotions"] == 3
    assert contract["hard_protocol"]["validation_wells"] == 45
    assert contract["hard_protocol"]["hard_density_wells"] == 27
    assert contract["derivative_gates"]["b11_closed_loop"][
        "maximum_median_relative_error"
    ] == 0.01
    assert contract["derivative_gates"]["b11_closed_loop"]["minimum_cosine"] == 0.99
    assert contract["test_policy"]["access_runs"] == 1
    assert contract["test_accessed"] is False
    assert json.loads(MANIFEST.read_text())["test_accessed"] is False
