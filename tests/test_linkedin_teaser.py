import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
teaser = importlib.import_module("scripts.build_linkedin_teaser")


def test_teaser_timeline_is_contiguous_and_exactly_27_seconds() -> None:
    assert teaser.SCENES[0][1] == 0.0
    assert teaser.SCENES[-1][2] == teaser.DURATION_SECONDS == 27
    assert all(
        left[2] == right[1]
        for left, right in zip(teaser.SCENES, teaser.SCENES[1:], strict=False)
    )
    assert teaser.FRAME_COUNT == teaser.FPS * teaser.DURATION_SECONDS == 810


def test_vertical_format_and_mobile_safe_margin_are_frozen() -> None:
    assert (teaser.WIDTH, teaser.HEIGHT, teaser.FPS) == (1080, 1350, 30)
    assert teaser.SAFE_MARGIN >= 72


def test_claim_copy_still_matches_frozen_validation_sources() -> None:
    claims = teaser.verify_claim_sources(teaser.load_assets())
    assert claims["before_pq"] == 0.4921920085941332
    assert claims["after_pq"] == 0.552294837627718
    assert claims["improved_fraction"] == 0.7469135802469136
    assert claims["causal_ci_lower_95"] > 0


def test_representative_frames_and_poster_render_at_delivery_size() -> None:
    assets = teaser.load_assets()
    for timestamp in (0.0, 4.5, 10.0, 15.5, 20.5, 24.5, 26.5):
        assert teaser.render_frame(timestamp, assets).size == (teaser.WIDTH, teaser.HEIGHT)
    assert teaser.render_poster(assets).size == (teaser.WIDTH, teaser.HEIGHT)
