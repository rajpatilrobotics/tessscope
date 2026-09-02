import importlib
import sys
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
fastcut = importlib.import_module("scripts.build_linkedin_fastcut")


def test_fastcut_uses_exact_uniform_timing_contract() -> None:
    assert fastcut.SPEED_FACTOR == Fraction(210, 29)
    assert fastcut.OUTPUT_DURATION_SECONDS == 29
    assert fastcut.FPS == 30
    assert fastcut.OUTPUT_FRAME_COUNT == 870
    assert fastcut.SETPTS_FILTER == "setpts=PTS*29/210,fps=fps=30:round=near:start_time=0"


def test_output_to_source_frame_mapping_spans_the_complete_source() -> None:
    assert fastcut.source_frame_for_output(0) == 0
    assert fastcut.source_frame_for_output(869) == 6293
    assert fastcut.mapped_source_seconds(869) == 869 / 30 * (210 / 29)


def test_render_command_contains_no_spatial_or_graphic_filter() -> None:
    command = fastcut.render_command("ffmpeg", PROJECT_ROOT / "derived.mp4")
    filter_value = command[command.index("-vf") + 1]
    assert filter_value == fastcut.SETPTS_FILTER
    assert all(token not in filter_value for token in ("crop", "scale", "overlay", "drawtext"))
    assert "-an" in command
