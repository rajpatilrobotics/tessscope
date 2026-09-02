import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
base = importlib.import_module("scripts.build_end_to_end_demo")
demo = importlib.import_module("scripts.build_end_to_end_demo_polished")


def test_polished_timeline_is_contiguous_and_exactly_29_seconds() -> None:
    assert demo.SCENES[0][1] == 0.0
    assert demo.SCENES[-1][2] == demo.DURATION_SECONDS == 29
    assert all(
        left[2] == right[1]
        for left, right in zip(demo.SCENES, demo.SCENES[1:], strict=False)
    )
    assert demo.FRAME_COUNT == 870


def test_polished_candidate_has_separate_paths() -> None:
    assert demo.DEFAULT_OUTPUT != base.DEFAULT_OUTPUT
    assert "polished" in demo.DEFAULT_OUTPUT.name
    assert demo.DEFAULT_OUTPUT not in demo.PRESERVED_VIDEO_PATHS
    assert base.DEFAULT_OUTPUT in demo.PRESERVED_VIDEO_PATHS


def test_preserved_video_hashes_match_frozen_values() -> None:
    records = demo.preserved_video_hashes()
    assert set(records) == set(demo.EXPECTED_PRESERVED_HASHES)
    assert all(record["preserved"] for record in records.values())


def test_claims_still_match_frozen_validation() -> None:
    assets = base.load_assets()
    assert base.verify_claim_sources(assets) == base.EXPECTED_CLAIMS


def test_all_scenes_transitions_and_psf_depths_render() -> None:
    assets = base.load_assets()
    timestamps = (
        0.0,
        2.9,
        3.6,
        5.9,
        6.9,
        10.8,
        11.9,
        15.6,
        15.9,
        18.7,
        20.9,
        23.0,
        24.9,
        26.5,
        26.9,
        28.2,
    )
    for timestamp in timestamps:
        assert demo.render_frame(timestamp, assets).size == (1920, 1080)
    for index in range(7):
        timestamp = 3.0 + (index + 0.25) * (4.0 / 7.0)
        assert demo.render_frame(timestamp, assets).size == (1920, 1080)


def test_locked_copy_has_required_claims_and_no_removed_marketing_phrases() -> None:
    copy = demo.COPY_PATH.read_text()
    assert "PQ 0.4922 → 0.5523" in copy
    assert "74.7% of hard frames improve" in copy
    assert "+0.0063 PQ" in copy
    assert "95% CI [+0.0007, +0.0115]" in copy
    assert "Tesseract Hackathon 2026 · Track 05" in copy
    renderer = Path(demo.__file__).read_text().lower()
    assert "not a live microscope feed" not in renderer
    assert "cached replay" not in renderer
    assert "validation-only" not in renderer
