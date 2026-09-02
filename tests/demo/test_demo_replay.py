"""Integrity tests for the compact validation replay evidence pack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tessscope.demo.evidence import sha256_array, sha256_path, write_deterministic_npz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "artifacts" / "runs" / "demo" / "validation-replay.json"
ARRAYS_PATH = PROJECT_ROOT / "artifacts" / "runs" / "demo" / "validation-replay.npz"


def _pointer(document: object, pointer: str) -> object:
    value = document
    for token in pointer.removeprefix("/").split("/"):
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def test_replay_archive_is_small_complete_and_hash_verified() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    assert metadata["status"] == "complete_cached_validation_replay"
    assert metadata["mode"] == "cached_replay_not_live_inference"
    assert metadata["split"] == "validation"
    assert metadata["test_accessed"] is False
    assert metadata["system_order"] == ["clear", "exact", "stopped", "piecewise"]
    assert metadata["corrected_system_order"] == ["exact", "stopped", "piecewise"]
    assert metadata["depths_um"] == [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
    assert metadata["archive"]["sha256"] == sha256_path(ARRAYS_PATH)
    assert metadata["archive"]["size_bytes"] == ARRAYS_PATH.stat().st_size
    assert ARRAYS_PATH.stat().st_size < 20 * 1024 * 1024

    with np.load(ARRAYS_PATH, allow_pickle=False) as archive:
        assert set(archive.files) == set(metadata["arrays"])
        for name, trace in metadata["arrays"].items():
            array = archive[name]
            assert str(array.dtype) == trace["dtype"]
            assert list(array.shape) == trace["shape"]
            assert sha256_array(array) == trace["sha256"]


def test_replay_archive_is_byte_deterministic(tmp_path: Path) -> None:
    regenerated = tmp_path / "validation-replay.npz"
    with np.load(ARRAYS_PATH, allow_pickle=False) as archive:
        write_deterministic_npz(regenerated, {name: archive[name] for name in archive.files})
    assert sha256_path(regenerated) == sha256_path(ARRAYS_PATH)


def test_replay_arrays_have_real_labels_and_physical_psfs() -> None:
    with np.load(ARRAYS_PATH, allow_pickle=False) as archive:
        assert archive["sensor_before"].shape == (4, 7, 256, 256)
        assert archive["sensor_corrected"].shape == (3, 7, 256, 256)
        assert archive["labels_before"].shape == (4, 7, 330, 330)
        assert archive["labels_corrected"].shape == (3, 7, 330, 330)
        assert archive["pupil_phase_radians"].shape == (4, 256, 256)
        assert archive["psf_sensor"].shape == (4, 7, 96, 96)
        assert np.count_nonzero(archive["target_labels"]) > 0
        assert np.count_nonzero(archive["labels_before"]) > 0
        assert np.count_nonzero(archive["labels_corrected"]) > 0
        assert np.all(archive["sensor_before"] >= 0.0)
        assert np.all(archive["sensor_corrected"] >= 0.0)
        assert np.all(archive["psf_sensor"] >= 0.0)
        np.testing.assert_allclose(archive["psf_sensor"].sum(axis=(-2, -1)), 1.0, atol=1e-6)
        outside = archive["pupil_mask"] == 0
        assert np.all(archive["pupil_phase_radians"][:, outside] == 0.0)


def test_frame_traceability_resolves_to_frozen_values() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    assert len(metadata["frames"]) == 21
    sources: dict[str, dict] = {}
    for frame in metadata["frames"]:
        source_path = PROJECT_ROOT / frame["source"]
        assert sha256_path(source_path) == frame["source_sha256"]
        document = sources.setdefault(frame["source"], json.loads(source_path.read_text()))
        before = _pointer(document, frame["before_pointer"])
        assert before["panoptic_quality"] == frame["before_pq"]
        assert before["predicted_depth_um"] == frame["predicted_depth_um"]
        if frame["depth_um"] != 0.0:
            corrected = _pointer(document, frame["corrected_pointer"])
            assert corrected["after_pq"] == frame["after_pq"]
            assert corrected["residual_depth_um"] == frame["residual_depth_um"]

    query = next(patch for patch in metadata["patches"] if patch["role"] == "query")
    assert query["field_id"] == "n21_s1"
    assert query["origin_yx"] == [0, 220]
    for patch in metadata["patches"]:
        assert patch["split"] == "validation"
        assert sha256_path(PROJECT_ROOT / patch["object_source"]) == patch[
            "object_source_sha256"
        ]
        assert sha256_path(PROJECT_ROOT / patch["label_source"]) == patch[
            "label_source_sha256"
        ]


def test_display_transform_is_global_and_matches_preregistration() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    transform = metadata["observer"]["transform"]
    assert transform == {
        "clip": [0.0, 1.0],
        "mode": "affine",
        "offset": 0.0,
        "per_image_normalization": False,
        "scale": 11.582016617246811,
    }
