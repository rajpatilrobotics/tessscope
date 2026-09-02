import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
demo = importlib.import_module("scripts.build_end_to_end_demo")


def test_timeline_is_contiguous_and_exactly_29_seconds() -> None:
    assert demo.SCENES[0][1] == 0.0
    assert demo.SCENES[-1][2] == demo.DURATION_SECONDS == 29
    assert all(
        left[2] == right[1]
        for left, right in zip(demo.SCENES, demo.SCENES[1:], strict=False)
    )
    assert demo.FRAME_COUNT == demo.FPS * demo.DURATION_SECONDS == 870


def test_landscape_delivery_contract_is_frozen() -> None:
    assert (demo.WIDTH, demo.HEIGHT, demo.FPS) == (1920, 1080, 30)
    assert demo.SAFE_MARGIN >= 72
    assert demo.SCENES[-1][2] - demo.SCENES[-1][1] == 2.0


def test_claims_match_frozen_validation_sources() -> None:
    assets = demo.load_assets()
    claims = demo.verify_claim_sources(assets)
    assert claims == demo.EXPECTED_CLAIMS


def test_each_scene_and_transition_render_at_delivery_size() -> None:
    assets = demo.load_assets()
    timestamps = (
        0.0,
        2.85,
        4.8,
        6.85,
        8.5,
        11.35,
        12.7,
        15.35,
        17.0,
        20.35,
        22.5,
        24.35,
        25.5,
        26.85,
        28.2,
    )
    for timestamp in timestamps:
        assert demo.render_frame(timestamp, assets).size == (demo.WIDTH, demo.HEIGHT)


def test_copy_contract_contains_required_positive_claims() -> None:
    copy = demo.COPY_PATH.read_text()
    assert "PQ 0.4922 → 0.5523" in copy
    assert "74.7% of hard frames improve" in copy
    assert "+0.0063 PQ" in copy
    assert "95% CI [+0.0007, +0.0115]" in copy
    assert "Forward-identical stopped control" in copy
