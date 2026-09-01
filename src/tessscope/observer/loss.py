"""Frozen differentiable design loss for the InstanSeg observer.

Ground-truth labels, object centers, validity flags, and crop locations remain
static. Gradients flow through the sensor image, frozen FCN, crop and center
embeddings, sigma fields, seed logits, and frozen pixel classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional

from tessscope.observer.instanseg import raw_head

MEMBERSHIP_CROP_PX = 64
MAXIMUM_INSTANCES = 32
SMOOTH_MAX_TEMPERATURE = 0.1
PHASE_L2_WEIGHT = 1e-4


@dataclass(frozen=True)
class ObserverTransform:
    """One validation-selected global transform with no per-image statistics."""

    mode: Literal["identity", "affine", "asinh"] = "identity"
    offset: float = 0.0
    scale: float = 1.0

    def apply(self, sensor: torch.Tensor) -> torch.Tensor:
        if self.scale <= 0:
            raise ValueError("Observer-transform scale must be positive")
        if self.mode == "identity":
            transformed = sensor
        else:
            normalized = (sensor - self.offset) / self.scale
            if self.mode == "affine":
                transformed = normalized
            elif self.mode == "asinh":
                transformed = torch.asinh(normalized)
            else:
                raise ValueError(f"Unknown observer transform: {self.mode}")
        return transformed.clamp(0.0, 1.0)


DEFAULT_OBSERVER_TRANSFORM = ObserverTransform()


def _binary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dimensions: tuple[int, ...],
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    return 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)


def _coordinate_map(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reproduce the frozen InstanSeg linear x/y coordinate convention."""
    xx = torch.linspace(0, width * 64 / 256, width, device=device, dtype=dtype)
    yy = torch.linspace(0, height * 64 / 256, height, device=device, dtype=dtype)
    return torch.stack(
        [
            xx.view(1, width).expand(height, width),
            yy.view(height, 1).expand(height, width),
        ]
    )


def _crop_grid(
    centers_yx: torch.Tensor,
    height: int,
    width: int,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed integer crop indices around the supplied GT centers."""
    half = crop_size // 2
    centers = torch.floor(centers_yx + 0.5).to(torch.long)
    center_y = centers[..., 0].clamp(min=half, max=height - half)
    center_x = centers[..., 1].clamp(min=half, max=width - half)
    offsets = torch.arange(-half, half, device=centers.device)
    grid_y = center_y[..., None, None] + offsets.view(1, 1, crop_size, 1)
    grid_x = center_x[..., None, None] + offsets.view(1, 1, 1, crop_size)
    grid_y = grid_y.expand(-1, -1, crop_size, crop_size)
    grid_x = grid_x.expand(-1, -1, crop_size, crop_size)
    crop_centers = torch.stack([center_y, center_x], dim=-1)
    return crop_centers, grid_y, grid_x


def _gather_crops(value: torch.Tensor, linear_indices: torch.Tensor) -> torch.Tensor:
    """Gather N,C,H,W features into N,K,S,S,C crops."""
    batch, channels = value.shape[:2]
    object_count, crop_height, crop_width = linear_indices.shape[1:]
    indices = linear_indices.reshape(batch, 1, -1).expand(-1, channels, -1)
    gathered = torch.gather(value.flatten(2), 2, indices)
    return gathered.reshape(
        batch, channels, object_count, crop_height, crop_width
    ).permute(0, 2, 3, 4, 1)


def _membership_loss(
    model: torch.nn.Module,
    spatial_embedding: torch.Tensor,
    sigma: torch.Tensor,
    labels: torch.Tensor,
    centers_yx: torch.Tensor,
    valid_objects: torch.Tensor,
    crop_size: int = MEMBERSHIP_CROP_PX,
) -> torch.Tensor:
    """Return one fixed-center instance loss per image/depth item."""
    batch, _, height, width = spatial_embedding.shape
    object_count = centers_yx.shape[1]
    centers, grid_y, grid_x = _crop_grid(centers_yx, height, width, crop_size)
    linear_indices = grid_y * width + grid_x

    embedding_crops = _gather_crops(spatial_embedding, linear_indices)
    sigma_crops = _gather_crops(sigma, linear_indices)

    center_indices = centers[..., 0] * width + centers[..., 1]
    center_embedding = torch.gather(
        spatial_embedding.flatten(2),
        2,
        center_indices[:, None].expand(-1, spatial_embedding.shape[1], -1),
    ).permute(0, 2, 1)
    features = torch.cat(
        [
            embedding_crops - center_embedding[:, :, None, None],
            sigma_crops,
        ],
        dim=-1,
    )
    membership_logits = model.pixel_classifier(features.reshape(-1, features.shape[-1]))
    membership_logits = membership_logits.reshape(
        batch, object_count, crop_size, crop_size
    )

    label_crops = torch.gather(
        labels.reshape(batch, -1), 1, linear_indices.reshape(batch, -1)
    ).reshape(batch, object_count, crop_size, crop_size)
    object_ids = torch.arange(
        1, object_count + 1, device=labels.device, dtype=labels.dtype
    ).view(1, object_count, 1, 1)
    target = (label_crops == object_ids).to(membership_logits.dtype)

    binary_cross_entropy = functional.binary_cross_entropy_with_logits(
        membership_logits, target, reduction="none"
    ).mean(dim=(-2, -1))
    dice = _binary_dice_loss(membership_logits, target, dimensions=(-2, -1))
    per_object = 0.5 * binary_cross_entropy + 0.5 * dice
    valid = valid_objects.to(per_object.dtype)
    return (per_object * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def design_loss_components(
    model: torch.nn.Module,
    sensor: torch.Tensor,
    instance_labels: torch.Tensor,
    centers_yx: torch.Tensor,
    valid_objects: torch.Tensor,
    transform: ObserverTransform = DEFAULT_OBSERVER_TRANSFORM,
) -> dict[str, torch.Tensor]:
    """Evaluate the frozen per-depth seed and fixed-center membership losses."""
    if sensor.ndim != 4:
        raise ValueError(f"Expected B,D,H,W sensor input, got {tuple(sensor.shape)}")
    if instance_labels.ndim != 3:
        raise ValueError("Instance labels must have shape B,H,W")
    batch, depth_count = sensor.shape[:2]
    if instance_labels.shape[0] != batch:
        raise ValueError("Sensor and instance-label batch sizes differ")
    if centers_yx.shape != (batch, MAXIMUM_INSTANCES, 2):
        raise ValueError(
            f"Centers must have shape B,{MAXIMUM_INSTANCES},2, got {centers_yx.shape}"
        )
    if valid_objects.shape != (batch, MAXIMUM_INSTANCES):
        raise ValueError(
            f"Validity must have shape B,{MAXIMUM_INSTANCES}, got {valid_objects.shape}"
        )

    observer_height, observer_width = instance_labels.shape[-2:]
    flattened_sensor = sensor.reshape(batch * depth_count, 1, *sensor.shape[-2:])
    resized = functional.interpolate(
        flattened_sensor,
        size=(observer_height, observer_width),
        mode="bilinear",
        align_corners=False,
    )
    transformed = transform.apply(resized)
    prediction = raw_head(model, transformed)
    if prediction.shape[1] != 5:
        raise ValueError(f"Frozen observer must return five channels, got {prediction.shape}")

    coordinate_map = _coordinate_map(
        observer_height, observer_width, prediction.device, prediction.dtype
    )
    spatial_embedding = (
        torch.sigmoid(prediction[:, :2]) - 0.5
    ) * 8.0 + coordinate_map[None]
    sigma = prediction[:, 2:4]
    seed_logits = prediction[:, 4:5]

    repeated_labels = instance_labels.repeat_interleave(depth_count, dim=0).to(
        prediction.device
    )
    repeated_centers = centers_yx.repeat_interleave(depth_count, dim=0).to(
        prediction.device
    )
    repeated_valid = valid_objects.repeat_interleave(depth_count, dim=0).to(
        prediction.device
    )
    foreground = (repeated_labels > 0).to(seed_logits.dtype)[:, None]

    seed_bce = functional.binary_cross_entropy_with_logits(seed_logits, foreground)
    seed_dice = _binary_dice_loss(
        seed_logits, foreground, dimensions=(-3, -2, -1)
    ).mean()
    seed_loss = 0.5 * seed_bce + 0.5 * seed_dice

    per_item_instance = _membership_loss(
        model,
        spatial_embedding,
        sigma,
        repeated_labels,
        repeated_centers,
        repeated_valid,
    )
    per_item_seed_bce = functional.binary_cross_entropy_with_logits(
        seed_logits, foreground, reduction="none"
    ).mean(dim=(-3, -2, -1))
    per_item_seed_dice = _binary_dice_loss(
        seed_logits, foreground, dimensions=(-3, -2, -1)
    )
    per_item_seed = 0.5 * per_item_seed_bce + 0.5 * per_item_seed_dice
    per_depth = (per_item_seed + 1.5 * per_item_instance).reshape(
        batch, depth_count
    ).mean(dim=0)
    return {
        "seed": seed_loss,
        "instance": per_item_instance.mean(),
        "per_depth": per_depth,
    }


def aggregate_depth_loss(
    per_depth: torch.Tensor,
    temperature: float = SMOOTH_MAX_TEMPERATURE,
) -> torch.Tensor:
    """Combine the average and smooth worst-depth terms from contract v2."""
    if per_depth.ndim != 1 or per_depth.numel() == 0:
        raise ValueError("Per-depth loss must be a non-empty vector")
    if temperature <= 0:
        raise ValueError("Smooth-max temperature must be positive")
    smooth_max = temperature * (
        torch.logsumexp(per_depth / temperature, dim=0)
        - torch.log(per_depth.new_tensor(float(per_depth.numel())))
    )
    return 0.75 * per_depth.mean() + 0.25 * smooth_max


def design_task_loss(
    model: torch.nn.Module,
    sensor: torch.Tensor,
    phase_coefficients: torch.Tensor,
    instance_labels: torch.Tensor,
    centers_yx: torch.Tensor,
    valid_objects: torch.Tensor,
    transform: ObserverTransform = DEFAULT_OBSERVER_TRANSFORM,
) -> torch.Tensor:
    """Return the scalar task objective, including frozen phase regularization."""
    components = design_loss_components(
        model,
        sensor,
        instance_labels,
        centers_yx,
        valid_objects,
        transform,
    )
    return aggregate_depth_loss(components["per_depth"]) + PHASE_L2_WEIGHT * (
        phase_coefficients.square().sum()
    )


def surrogate_proxy_loss(
    sensor: torch.Tensor,
    instance_labels: torch.Tensor,
    transform: ObserverTransform = DEFAULT_OBSERVER_TRANSFORM,
    logit_scale: float = 8.0,
) -> torch.Tensor:
    """Approved label-aware proxy used only for the surrogate backward path."""
    if sensor.ndim != 4:
        raise ValueError("Expected B,D,H,W sensor input")
    if instance_labels.ndim != 3 or instance_labels.shape[0] != sensor.shape[0]:
        raise ValueError("Instance labels must have matching B,H,W shape")
    if logit_scale <= 0:
        raise ValueError("Proxy logit scale must be positive")
    batch, depth_count = sensor.shape[:2]
    flattened = sensor.reshape(batch * depth_count, 1, *sensor.shape[-2:])
    resized = functional.interpolate(
        flattened,
        size=instance_labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    transformed = transform.apply(resized)
    foreground = (instance_labels > 0).to(transformed.dtype)[:, None]
    foreground = foreground.repeat_interleave(depth_count, dim=0)
    logits = logit_scale * (transformed - 0.5)
    binary_cross_entropy = functional.binary_cross_entropy_with_logits(
        logits, foreground
    )
    dice = _binary_dice_loss(
        logits, foreground, dimensions=(-3, -2, -1)
    ).mean()
    return 0.5 * binary_cross_entropy + 0.5 * dice


def official_hard_labels(
    model: torch.nn.Module,
    sensor: torch.Tensor,
    observer_shape: tuple[int, int],
    transform: ObserverTransform = DEFAULT_OBSERVER_TRANSFORM,
) -> torch.Tensor:
    """Run the bundled official hard endpoint after frozen observer preprocessing."""
    if sensor.ndim != 4:
        raise ValueError("Expected B,D,H,W sensor input")
    flattened = sensor.reshape(-1, 1, *sensor.shape[-2:])
    resized = functional.interpolate(
        flattened, size=observer_shape, mode="bilinear", align_corners=False
    )
    return model(transform.apply(resized))
