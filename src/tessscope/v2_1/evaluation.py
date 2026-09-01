"""Well-paired hard-validation helpers for TessScope v2.1."""

from __future__ import annotations

import numpy as np

from tessscope.evaluation.metrics import grouped_bootstrap_difference


def paired_hard_pq_bootstrap(
    all_rows: list[dict],
    candidate_name: str,
    reference_name: str,
) -> dict:
    """Bootstrap paired hard-density off-focus PQ differences by whole well."""
    candidate = {
        (row["well"], row["field_id"], row["depth_um"]): row
        for row in all_rows
        if row["design"] == candidate_name
        and row["hard_dense_patch"]
        and row["depth_um"] != 0.0
    }
    reference = {
        (row["well"], row["field_id"], row["depth_um"]): row
        for row in all_rows
        if row["design"] == reference_name
        and row["hard_dense_patch"]
        and row["depth_um"] != 0.0
    }
    if candidate.keys() != reference.keys() or not candidate:
        raise ValueError("Candidate/reference hard rows are not paired")
    keys = sorted(candidate)
    result = grouped_bootstrap_difference(
        np.asarray([key[0] for key in keys]),
        np.asarray([candidate[key]["panoptic_quality"] for key in keys]),
        np.asarray([reference[key]["panoptic_quality"] for key in keys]),
        replicates=2000,
        seed=20260901,
    )
    result["candidate"] = candidate_name
    result["reference"] = reference_name
    return result
