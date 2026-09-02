"""Output and numerical-integrity tests for the demo figure pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tessscope.demo.evidence import sha256_path
from tessscope.demo.figures import bootstrap_mean_interval, display_sensor, overlay_boundaries

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "demo" / "figure-manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_global_sensor_transform_does_not_use_image_statistics() -> None:
    sensor = np.asarray([[0.0, 5.0], [10.0, 20.0]], dtype=np.float32)
    transformed = display_sensor(sensor, scale=10.0)
    np.testing.assert_array_equal(transformed, [[0.0, 0.5], [1.0, 1.0]])

    target = np.asarray([[0, 0], [1, 1]], dtype=np.int32)
    prediction = np.asarray([[0, 2], [0, 2]], dtype=np.int32)
    overlay = overlay_boundaries(sensor, target, prediction, scale=10.0)
    assert overlay.shape == (2, 2, 3)
    assert np.all((overlay >= 0.0) & (overlay <= 1.0))


def test_grouped_bootstrap_is_deterministic() -> None:
    first = bootstrap_mean_interval([0.1, 0.2, 0.4, 0.8], seed_key="demo-test")
    second = bootstrap_mean_interval([0.1, 0.2, 0.4, 0.8], seed_key="demo-test")
    assert first == second
    assert first[0] == pytest.approx(0.375)
    assert first[1] <= first[0] <= first[2]


def test_publication_outputs_exist_in_all_required_formats() -> None:
    manifest = _manifest()
    assert manifest["status"] == "complete_validation_only_publication_outputs"
    assert manifest["test_accessed"] is False
    assert set(manifest["figures"]) == {
        "causal-comparison",
        "gradient-architecture",
        "hero-evidence",
        "matched-microscopy",
        "pq-focus-depth",
        "pupil-psf-depth",
    }
    for figure in manifest["figures"].values():
        assert len(figure["caption"]) > 80
        assert len(figure["alt"]) > 80
        assert set(figure["files"]) == {"png", "svg", "pdf"}
        for suffix, output in figure["files"].items():
            path = PROJECT_ROOT / output["path"]
            assert path.suffix == f".{suffix}"
            assert output["sha256"] == sha256_path(path)
            assert output["size_bytes"] == path.stat().st_size
            assert path.stat().st_size > 10_000
        with Image.open(PROJECT_ROOT / figure["files"]["png"]["path"]) as image:
            assert image.width >= 3000
            assert image.height >= 1200
            assert image.info["dpi"][0] == pytest.approx(300, abs=1)


def test_hero_and_architecture_preserve_scope_and_runtime_boundaries() -> None:
    figures = _manifest()["figures"]
    hero = figures["hero-evidence"]
    assert hero["field_id"] == "n21_s1"
    assert hero["depth_um"] == -2.0
    assert hero["before_pq"] == pytest.approx(0.6294106178269091)
    assert hero["after_pq"] == pytest.approx(0.6302726038728452)
    assert hero["display_transform"]["per_image_normalization"] is False
    assert set(hero["replay_arrays"]) == {
        "labels_before",
        "labels_corrected",
        "psf_sensor",
        "pupil_mask",
        "pupil_phase_radians",
        "sensor_before",
        "sensor_corrected",
        "target_labels",
    }

    architecture = figures["gradient-architecture"]
    assert architecture["component_boundaries"] == [
        {"component": "optics", "runtime": "JAX/Chromatix", "served_calls": 2},
        {"component": "autofocus", "runtime": "NumPy/SciPy", "served_calls": 1},
        {"component": "observer", "runtime": "PyTorch/InstanSeg", "served_calls": 1},
    ]


def test_causal_figure_preserves_supported_and_unsupported_results() -> None:
    causal = _manifest()["figures"]["causal-comparison"]
    assert causal["before_pq"] == pytest.approx(0.4921920085941332)
    assert causal["after_pq"] == pytest.approx(0.552294837627718)
    assert causal["improved_fraction"] == pytest.approx(0.7469135802469136)
    by_label = {row["label"]: row for row in causal["comparisons"]}
    stopped = by_label["Exact − stopped gradient"]
    assert stopped["mean_difference"] == pytest.approx(0.006258840610359453)
    assert stopped["ci_lower_95"] > 0.0
    assert stopped["status"] == "supported"
    piecewise = by_label["Exact − piecewise-028"]
    assert piecewise["mean_difference"] == pytest.approx(0.0012850136904830492)
    assert piecewise["ci_lower_95"] < 0.0 < piecewise["ci_upper_95"]
    assert piecewise["status"] == "not significant"


def test_depth_curves_are_grouped_by_well_and_complete() -> None:
    figure = _manifest()["figures"]["pq-focus-depth"]
    for curve in (*figure["pq_curves"].values(), *figure["focus_curves"].values()):
        assert len(curve) == 6
        assert [row["depth_um"] for row in curve] == [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0]
        assert {row["well_count"] for row in curve} == {27}
        assert {row["bootstrap_replicates"] for row in curve} == {2000}
        for row in curve:
            assert row["ci_lower_95"] <= row["mean"] <= row["ci_upper_95"]


def test_psf_rendering_and_animation_use_fixed_complete_grids() -> None:
    manifest = _manifest()
    psf = manifest["figures"]["pupil-psf-depth"]["psf_display"]
    assert psf["source_shape_px"] == [96, 96]
    assert psf["crop_shape_px"] == [41, 41]
    assert psf["limits"] == [-5.0, 0.0]

    replay = manifest["replay"]
    assert len(replay["frames"]) == 7
    assert [row["depth_um"] for row in replay["frames"]] == [
        -6.0,
        -4.0,
        -2.0,
        0.0,
        2.0,
        4.0,
        6.0,
    ]
    for frame in replay["frames"]:
        path = PROJECT_ROOT / frame["path"]
        assert frame["sha256"] == sha256_path(path)
        assert len(frame["traced_frames"]) == 3
    gif_path = PROJECT_ROOT / replay["gif"]["path"]
    mp4_path = PROJECT_ROOT / replay["mp4"]["path"]
    assert gif_path.read_bytes().startswith(b"GIF89a")
    assert b"ftyp" in mp4_path.read_bytes()[:32]
    with Image.open(gif_path) as image:
        assert image.n_frames == 7
    assert replay["gif"]["sha256"] == sha256_path(gif_path)
    assert replay["mp4"]["sha256"] == sha256_path(mp4_path)


def test_figure_manifest_is_byte_stable_for_frozen_outputs() -> None:
    assert sha256_path(MANIFEST_PATH) == (
        "4b9185e8115232aaf0250034c59160b9bcb8d4d6f21eb89f85f24fa40c121453"
    )
