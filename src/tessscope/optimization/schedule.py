"""One frozen patch/depth schedule shared by all learned designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessscope.data.bbbc039 import patch_origins, records_for_split

DEPTHS_UM = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)


@dataclass(frozen=True)
class PatchKey:
    image_id: str
    origin_yx: tuple[int, int]


@dataclass(frozen=True)
class BatchSpec:
    step: int
    patches: tuple[PatchKey, ...]
    depth_indices: tuple[int, int, int]
    depths_um: tuple[float, float, float]


def training_patch_keys() -> list[PatchKey]:
    """Return the sorted 300-patch decontaminated training catalog."""
    return [
        PatchKey(record.image_id, origin)
        for record in records_for_split("training")
        for origin in patch_origins("training")
    ]


def build_training_schedule(
    *,
    steps: int = 120,
    batch_size: int = 2,
    seed: int = 20260901,
) -> list[BatchSpec]:
    """Create deterministic permutations and the frozen cyclic depth triples."""
    if steps <= 0 or batch_size <= 0:
        raise ValueError("Schedule steps and batch size must be positive")
    catalog = training_patch_keys()
    generator = np.random.default_rng(seed)
    needed = steps * batch_size
    ordered: list[PatchKey] = []
    while len(ordered) < needed:
        ordered.extend(catalog[index] for index in generator.permutation(len(catalog)))
    ordered = ordered[:needed]
    result = []
    for step in range(steps):
        indices = tuple((step + offset) % len(DEPTHS_UM) for offset in (0, 2, 4))
        result.append(
            BatchSpec(
                step=step,
                patches=tuple(ordered[step * batch_size : (step + 1) * batch_size]),
                depth_indices=indices,
                depths_um=tuple(DEPTHS_UM[index] for index in indices),
            )
        )
    return result
