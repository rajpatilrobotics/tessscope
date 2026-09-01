"""Exact access to the frozen InstanSeg raw head and hard postprocessor.

The release TorchScript exposes its traced ``fcn`` as a child module.  The
official Python postprocessor is used only for raw-head parity and diagnostics;
headline evaluation calls the bundled TorchScript hard endpoint directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_BUNDLE_DIR = PROJECT_ROOT / "artifacts" / "external" / "instanseg" / "model-v0.1.2"
MODEL_PATH = MODEL_BUNDLE_DIR / "instanseg.pt"


@dataclass(frozen=True)
class InstanSegSemantics:
    """Model-specific channel meanings and frozen hard-postprocess settings."""

    dim_coords: int
    n_sigma: int
    dim_seeds: int
    pixel_size_um: float
    min_size: int
    mask_threshold: float
    peak_distance: int
    seed_threshold: float
    overlap_threshold: float
    mean_threshold: float
    fg_threshold: float
    window_size: int
    cleanup_fragments: bool

    @property
    def channels_per_target(self) -> int:
        """Return raw channels ordered as coordinates, sigma, then seed."""
        return self.dim_coords + self.n_sigma + self.dim_seeds

    @classmethod
    def from_model(cls, model: torch.jit.ScriptModule) -> InstanSegSemantics:
        """Read values serialized in the official TorchScript wrapper."""
        return cls(
            dim_coords=int(model.dim_coords),
            n_sigma=int(model.n_sigma),
            dim_seeds=int(model.dim_seeds),
            pixel_size_um=float(model.pixel_size),
            min_size=int(model.default_min_size),
            mask_threshold=float(model.default_mask_threshold),
            peak_distance=int(model.default_peak_distance),
            seed_threshold=float(model.default_seed_threshold),
            overlap_threshold=float(model.default_overlap_threshold),
            mean_threshold=float(model.default_mean_threshold),
            fg_threshold=float(model.default_fg_threshold),
            window_size=int(model.default_window_size),
            cleanup_fragments=bool(model.default_cleanup_fragments),
        )


def load_frozen_model(
    device: str | torch.device = "cpu", model_path: Path = MODEL_PATH
) -> torch.jit.ScriptModule:
    """Load the exact release model, freeze weights, and retain input gradients."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing InstanSeg model at {model_path}. Run scripts/fetch_instanseg.py first."
        )
    model = torch.jit.load(str(model_path), map_location=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def normalize_release_input(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply the model RDF's 0.1/99.9 percentile scale without clipping.

    The lack of clipping is important: clipping to [0, 1] changes 11 pixels in
    the supplied reference output even though it sounds like harmless cleanup.
    """
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 4:
        raise ValueError(f"Expected B,C,Y,X input, got shape {value.shape}")
    low, high = np.percentile(value, [0.1, 99.9], axis=(-2, -1), keepdims=True)
    scale = np.maximum(eps, high - low)
    return ((value - low) / scale).astype(np.float32)


def raw_head(model: torch.jit.ScriptModule, sensor: torch.Tensor) -> torch.Tensor:
    """Run the exposed frozen FCN with the wrapper's clamp/padding semantics."""
    from instanseg.utils.tiling import _instanseg_padding, _recover_padding

    if sensor.ndim != 4:
        raise ValueError(f"Expected B,C,Y,X sensor tensor, got {tuple(sensor.shape)}")
    clamped = sensor.clamp(min=-2.0, max=3.0)
    padded, pad = _instanseg_padding(clamped, extra_pad=0)
    prediction = model.fcn(padded)
    recovered = [_recover_padding(prediction[index], pad) for index in range(prediction.shape[0])]
    return torch.stack(recovered)


def reconstruct_hard_labels(
    model: torch.jit.ScriptModule, prediction: torch.Tensor
) -> torch.Tensor:
    """Reconstruct hard instances from raw output using official source code.

    Label IDs may differ from the TorchScript wrapper because sparse connected
    components are enumerated differently. The instance partition is checked
    after deterministic relabeling by :func:`canonicalize_labels`.
    """
    from instanseg.utils.loss.instanseg_loss import InstanSeg

    semantics = InstanSegSemantics.from_model(model)
    if prediction.ndim != 4 or prediction.shape[1] != semantics.channels_per_target:
        raise ValueError(
            "Expected B,C,Y,X raw head with "
            f"C={semantics.channels_per_target}, got {tuple(prediction.shape)}"
        )

    device = prediction.device
    method = InstanSeg(
        n_sigma=semantics.n_sigma,
        device=str(device),
        cells_and_nuclei=False,
        feature_engineering_function="0",
        dim_coords=semantics.dim_coords,
        dim_seeds=semantics.dim_seeds,
    )
    method.pixel_classifier = model.pixel_classifier

    labels: list[torch.Tensor] = []
    for sample in prediction:
        label = method.postprocessing(
            sample,
            mask_threshold=semantics.mask_threshold,
            peak_distance=semantics.peak_distance,
            seed_threshold=semantics.seed_threshold,
            overlap_threshold=semantics.overlap_threshold,
            mean_threshold=semantics.mean_threshold,
            fg_threshold=semantics.fg_threshold,
            window_size=semantics.window_size,
            min_size=semantics.min_size,
            device=str(device),
            cleanup_fragments=semantics.cleanup_fragments,
        )
        labels.append(label)
    return torch.stack(labels)


def canonicalize_labels(labels: np.ndarray | torch.Tensor) -> np.ndarray:
    """Relabel instances by their first row-major pixel for deterministic comparison."""
    if isinstance(labels, torch.Tensor):
        value = labels.detach().cpu().numpy()
    else:
        value = np.asarray(labels)
    result = np.zeros(value.shape, dtype=np.int32)
    ordered: list[tuple[int, Any]] = []
    flat = value.reshape(-1)
    for label in np.unique(value):
        if label == 0:
            continue
        ordered.append((int(np.flatnonzero(flat == label)[0]), label))
    for canonical_id, (_, label) in enumerate(sorted(ordered), start=1):
        result[value == label] = canonical_id
    return result
