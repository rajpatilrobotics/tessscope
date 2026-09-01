"""Frozen-checkpoint selection helpers for TessScope v2.4."""

from tessscope.v2_4.checkpoints import (
    FIRST_SEGMENTATION_MAXIMUM,
    eligible_intervals,
    is_soft_eligible,
    select_checkpoints,
)

__all__ = [
    "FIRST_SEGMENTATION_MAXIMUM",
    "eligible_intervals",
    "is_soft_eligible",
    "select_checkpoints",
]
