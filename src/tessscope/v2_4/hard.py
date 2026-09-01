"""Paired corrected-frame and stability evidence for v2.4 hard validation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from tessscope.evaluation.metrics import grouped_bootstrap_difference


def paired_corrected_pq_bootstrap(
    candidate_rows: list[dict],
    reference_rows: list[dict],
    candidate_name: str,
    reference_name: str,
) -> dict:
    """Bootstrap paired corrected-frame PQ differences by whole well."""
    candidate = {
        (row["well"], row["field_id"], row["original_depth_um"]): row
        for row in candidate_rows
        if row["design"] == candidate_name and row["original_depth_um"] != 0.0
    }
    reference = {
        (row["well"], row["field_id"], row["original_depth_um"]): row
        for row in reference_rows
        if row["design"] == reference_name and row["original_depth_um"] != 0.0
    }
    if candidate.keys() != reference.keys() or not candidate:
        raise ValueError("Candidate/reference corrected rows are not paired")
    keys = sorted(candidate)
    result = grouped_bootstrap_difference(
        np.asarray([key[0] for key in keys]),
        np.asarray([candidate[key]["after_pq"] for key in keys]),
        np.asarray([reference[key]["after_pq"] for key in keys]),
        replicates=2000,
        seed=20260901,
    )
    result["candidate"] = candidate_name
    result["reference"] = reference_name
    return result


def well_stability(
    candidate_rows: list[dict],
    reference_rows: list[dict],
    *,
    candidate_name: str,
    reference_name: str,
    depth_key: str,
    metric_key: str,
) -> dict:
    """Require majority-positive wells and a positive leave-one-well-out mean."""
    candidate = {
        (row["well"], row["field_id"], row[depth_key]): row
        for row in candidate_rows
        if row["design"] == candidate_name and row[depth_key] != 0.0
    }
    reference = {
        (row["well"], row["field_id"], row[depth_key]): row
        for row in reference_rows
        if row["design"] == reference_name and row[depth_key] != 0.0
    }
    if candidate.keys() != reference.keys() or not candidate:
        raise ValueError("Candidate/reference stability rows are not paired")
    by_well = defaultdict(list)
    for key in sorted(candidate):
        by_well[key[0]].append(
            float(candidate[key][metric_key]) - float(reference[key][metric_key])
        )
    well_means = {
        well: float(np.mean(differences)) for well, differences in by_well.items()
    }
    wells = sorted(well_means)
    leave_one_out = {
        omitted: float(np.mean([well_means[well] for well in wells if well != omitted]))
        for omitted in wells
    }
    positive_fraction = float(np.mean([well_means[well] > 0.0 for well in wells]))
    minimum_leave_one_out = min(leave_one_out.values())
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "well_count": len(wells),
        "well_mean_differences": well_means,
        "positive_well_fraction": positive_fraction,
        "minimum_required_positive_well_fraction": 0.60,
        "leave_one_well_out_mean_differences": leave_one_out,
        "minimum_leave_one_well_out_mean_difference": minimum_leave_one_out,
        "passed": bool(positive_fraction >= 0.60 and minimum_leave_one_out > 0.0),
    }
