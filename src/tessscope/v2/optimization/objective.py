"""Scale-transparent joint objective used across exact and ablation designs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointObjectiveWeights:
    """Nonnegative branch weights with focus normalized by the stage range."""

    segmentation: float = 1.0
    focus: float = 1.0
    maximum_depth_um: float = 6.0

    def validate(self) -> None:
        if self.segmentation < 0 or self.focus < 0:
            raise ValueError("Joint objective weights must be nonnegative")
        if self.segmentation + self.focus <= 0:
            raise ValueError("At least one joint objective branch must be active")
        if self.maximum_depth_um <= 0:
            raise ValueError("Maximum autofocus depth must be positive")

    def combine(self, segmentation_loss, focus_mse):
        """Combine branches while keeping a six-micron error on order one."""
        self.validate()
        return (
            self.segmentation * segmentation_loss
            + self.focus * focus_mse / (self.maximum_depth_um**2)
        )
