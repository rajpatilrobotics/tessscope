import numpy as np
from PIL import Image

from tessscope.data.decontam import (
    canonical_gray,
    d4_views,
    hamming_hex,
    normalized_view,
    perceptual_hashes,
)


def test_d4_views_cover_rotations_and_reflections() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    views = d4_views(image)
    assert len(views) == 8
    assert np.array_equal(views[1], np.rot90(image))
    assert np.array_equal(views[4], np.fliplr(image))


def test_normalized_view_has_unit_norm() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    view = normalized_view(image)
    assert abs(float(view.mean())) < 1e-6
    assert abs(float(np.linalg.norm(view)) - 1.0) < 1e-6


def test_perceptual_hash_is_rotation_robust() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[5:20, 9:15] = 1
    left = perceptual_hashes(image)
    right = perceptual_hashes(np.rot90(image))
    assert min(hamming_hex(a, b) for a in left for b in right) == 0


def test_canonical_gray_collapses_identical_rgb(tmp_path) -> None:
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    rgb = np.stack([gray, gray, gray], axis=-1)
    path = tmp_path / "image.png"
    Image.fromarray(rgb).save(path)
    assert np.array_equal(canonical_gray(path), gray.astype(np.float32))
