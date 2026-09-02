"""Verify the canonical TessScope 29-second end-to-end demo deliverables."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import build_end_to_end_demo as demo
from PIL import Image

QA_PATH = demo.OUTPUT_ROOT / "tessscope-end-to-end-demo-qa.json"


def _probe(ffprobe: str, path: Path) -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        cwd=demo.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _filter_scan(ffmpeg: str, path: Path) -> dict:
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(path),
            "-vf",
            "blackdetect=d=0.20:pix_th=0.02,freezedetect=n=-60dB:d=3.0",
            "-an",
            "-f",
            "null",
            "-",
        ],
        cwd=demo.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    black_intervals = re.findall(r"black_start:[^\n]+", result.stderr)
    freeze_durations = [
        float(value)
        for value in re.findall(r"freeze_duration: ([0-9]+(?:\.[0-9]+)?)", result.stderr)
    ]
    if black_intervals:
        raise ValueError(f"Unexpected black intervals: {black_intervals}")
    if any(value > 3.05 for value in freeze_durations):
        raise ValueError(f"Unexpected long static interval: {freeze_durations}")
    return {
        "black_intervals": black_intervals,
        "freeze_durations_seconds": freeze_durations,
        "long_freeze_threshold_seconds": 3.0,
    }


def _full_decode(ffmpeg: str, path: Path) -> None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-i", str(path), "-f", "null", "-"],
        cwd=demo.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode or result.stderr.strip():
        raise ValueError(f"Full decode failed: {result.stderr.strip()}")


def _verify_fast_start(path: Path) -> None:
    payload = path.read_bytes()
    moov = payload.find(b"moov")
    mdat = payload.find(b"mdat")
    if moov < 0 or mdat < 0 or moov > mdat:
        raise ValueError("Video is not fast-start optimized")


def _verify_artifact_dimensions() -> dict:
    expected = {
        demo.POSTER_PATH: (1920, 1080),
        demo.CONTACT_SHEET_PATH: (5760, 3240),
        demo.READABILITY_SHEET_PATH: (1920, 1080),
    }
    results = {}
    for path, expected_size in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            actual = image.size
        if actual != expected_size:
            raise ValueError(f"Unexpected dimensions for {path.name}: {actual}")
        results[str(path.relative_to(demo.PROJECT_ROOT))] = {
            "dimensions": list(actual),
            "sha256": demo.sha256_path(path),
        }
    return results


def _verify_documents() -> dict:
    for path in (demo.STORYBOARD_PATH, demo.COPY_PATH, demo.MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    copy = demo.COPY_PATH.read_text()
    required_copy = (
        "Phase pupil → PSF → first exposure",
        "Three Tesseracts · three native runtimes · one composed workflow.",
        "Exact gradient crosses every boundary.",
        "Seven phase parameters optimized end-to-end.",
        "PQ 0.4922 → 0.5523",
        "74.7% of hard frames improve",
        "+0.0063 PQ",
        "95% CI [+0.0007, +0.0115]",
        "Forward-identical stopped control",
    )
    missing = [value for value in required_copy if value not in copy]
    if missing:
        raise ValueError(f"Locked copy is missing: {missing}")
    return {
        str(path.relative_to(demo.PROJECT_ROOT)): demo.sha256_path(path)
        for path in (demo.STORYBOARD_PATH, demo.COPY_PATH, demo.MANIFEST_PATH)
    }


def _verify_preservation(manifest: dict) -> dict:
    expected = manifest["preserved_videos"]
    actual = demo.preserved_video_hashes()
    if actual != expected:
        raise ValueError("An earlier video changed after the end-to-end render")
    return actual


def verify(video: Path, compare: Path | None) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("FFmpeg and ffprobe are required")
    if not video.is_file():
        raise FileNotFoundError(video)
    if not demo.MANIFEST_PATH.is_file():
        raise FileNotFoundError(demo.MANIFEST_PATH)

    metadata = _probe(ffprobe, video)
    video_streams = [
        stream for stream in metadata["streams"] if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in metadata["streams"] if stream.get("codec_type") == "audio"
    ]
    if len(video_streams) != 1:
        raise ValueError("Expected exactly one video stream")
    stream = video_streams[0]
    expected_stream = {
        "codec_name": "h264",
        "width": demo.WIDTH,
        "height": demo.HEIGHT,
        "pix_fmt": "yuv420p",
        "r_frame_rate": "30/1",
        "nb_read_frames": str(demo.FRAME_COUNT),
    }
    if any(stream.get(key) != value for key, value in expected_stream.items()):
        raise ValueError(f"Unexpected video stream: {stream}")
    if audio_streams:
        raise ValueError("Muted-first demo must contain no audio stream")
    duration = float(metadata["format"]["duration"])
    if not math.isclose(duration, 29.0, rel_tol=0.0, abs_tol=1 / demo.FPS):
        raise ValueError(f"Unexpected duration: {duration}")

    _verify_fast_start(video)
    _full_decode(ffmpeg, video)
    scan = _filter_scan(ffmpeg, video)
    artifacts = _verify_artifact_dimensions()
    documents = _verify_documents()
    assets = demo.load_assets()
    claims = demo.verify_claim_sources(assets)
    for timestamp in (0.0, 3.4, 8.5, 12.5, 17.5, 22.5, 25.5, 28.2):
        if demo.render_frame(timestamp, assets).size != (demo.WIDTH, demo.HEIGHT):
            raise ValueError(f"Layout render failed at {timestamp:.2f}s")

    manifest = json.loads(demo.MANIFEST_PATH.read_text())
    if manifest["outputs"]["video"]["sha256"] != demo.sha256_path(video):
        raise ValueError("Manifest video hash does not match the encoded video")
    preservation = _verify_preservation(manifest)

    stability = {"checked": False, "byte_identical": None}
    if compare is not None:
        comparison = demo.project_path(compare)
        if not comparison.is_file():
            raise FileNotFoundError(comparison)
        canonical_hash = demo.sha256_path(video)
        comparison_hash = demo.sha256_path(comparison)
        if canonical_hash != comparison_hash:
            raise ValueError("Deterministic comparison render is not byte-identical")
        stability = {
            "checked": True,
            "byte_identical": True,
            "comparison_path": str(comparison.relative_to(demo.PROJECT_ROOT)),
            "sha256": canonical_hash,
        }

    return {
        "schema_version": 1,
        "status": "PASS_review_candidate",
        "selected_or_published": False,
        "video": {
            "path": str(video.relative_to(demo.PROJECT_ROOT)),
            "sha256": demo.sha256_path(video),
            "bytes": video.stat().st_size,
            "duration_seconds": duration,
            "frame_count": demo.FRAME_COUNT,
            "video_stream": stream,
            "audio_stream_count": 0,
            "fast_start": True,
            "full_decode_errors": 0,
        },
        "filter_scan": scan,
        "layout": {
            "delivery_dimensions": [demo.WIDTH, demo.HEIGHT],
            "minimum_supporting_copy_px": 26,
            "primary_supporting_copy_px": 30,
            "readability_review_tile_dimensions": [640, 360],
            "representative_frames_rendered_without_bounds_errors": True,
        },
        "claims": claims,
        "source_preservation": preservation,
        "artifacts": artifacts,
        "documents": documents,
        "deterministic_regeneration": stability,
        "manual_visual_review": {
            "status": "pending_user_or_independent_reviewer",
            "full_resolution_contact_sheet": str(
                demo.CONTACT_SHEET_PATH.relative_to(demo.PROJECT_ROOT)
            ),
            "readability_contact_sheet": str(
                demo.READABILITY_SHEET_PATH.relative_to(demo.PROJECT_ROOT)
            ),
        },
        "external_actions_performed": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=demo.DEFAULT_OUTPUT)
    parser.add_argument("--compare", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = demo.project_path(args.video)
    report = verify(video, args.compare)
    if video == demo.DEFAULT_OUTPUT.resolve():
        QA_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
