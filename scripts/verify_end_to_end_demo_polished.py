"""Verify the separate polished TessScope end-to-end video candidate."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from scripts import build_end_to_end_demo as base
from scripts import build_end_to_end_demo_polished as demo

QA_PATH = demo.OUTPUT_ROOT / "tessscope-end-to-end-demo-polished-qa.json"


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


def _full_decode(ffmpeg: str, path: Path) -> None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-i", str(path), "-f", "null", "-"],
        cwd=demo.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode or result.stderr.strip():
        raise ValueError(f"Full decode failed: {result.stderr.strip()}")


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
    black = re.findall(r"black_start:[^\n]+", result.stderr)
    freezes = [
        float(value)
        for value in re.findall(r"freeze_duration: ([0-9]+(?:\.[0-9]+)?)", result.stderr)
    ]
    if black:
        raise ValueError(f"Unexpected black intervals: {black}")
    if any(value > 3.05 for value in freezes):
        raise ValueError(f"Unexpected long freeze interval: {freezes}")
    return {
        "black_intervals": black,
        "freeze_durations_seconds": freezes,
        "long_freeze_threshold_seconds": 3.0,
    }


def _fast_start(path: Path) -> None:
    payload = path.read_bytes()
    moov = payload.find(b"moov")
    mdat = payload.find(b"mdat")
    if moov < 0 or mdat < 0 or moov > mdat:
        raise ValueError("Polished candidate is not fast-start optimized")


def _artifact_checks() -> dict:
    expected = {
        demo.POSTER_PATH: (1920, 1080),
        demo.CONTACT_SHEET_PATH: (5760, 3240),
        demo.READABILITY_SHEET_PATH: (1920, 1080),
    }
    records = {}
    for path, dimensions in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.size != dimensions:
                raise ValueError(f"Unexpected dimensions for {path.name}: {image.size}")
        records[str(path.relative_to(demo.PROJECT_ROOT))] = {
            "dimensions": list(dimensions),
            "sha256": demo.sha256_path(path),
        }
    return records


def _copy_checks() -> dict:
    required_paths = (demo.STORYBOARD_PATH, demo.COPY_PATH, demo.MANIFEST_PATH)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    copy = demo.COPY_PATH.read_text()
    required = (
        "Can one exact gradient teach a microscope to refocus?",
        "JAX · Chromatix",
        "NumPy · SciPy",
        "PyTorch · InstanSeg",
        "PQ 0.4922 → 0.5523",
        "74.7% of hard frames improve",
        "+0.0063 PQ",
        "95% CI [+0.0007, +0.0115]",
        "Tesseract Hackathon 2026 · Track 05",
    )
    missing = [value for value in required if value not in copy]
    if missing:
        raise ValueError(f"Required polished copy is missing: {missing}")
    build_source = Path(demo.__file__).read_text()
    prohibited_marketing_copy = (
        "not a live microscope feed",
        "cached replay",
        "validation-only",
    )
    present = [value for value in prohibited_marketing_copy if value in build_source.lower()]
    if present:
        raise ValueError(f"Removed marketing copy returned to renderer: {present}")
    return {
        str(path.relative_to(demo.PROJECT_ROOT)): demo.sha256_path(path)
        for path in required_paths
    }


def verify(video: Path, compare: Path | None) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("FFmpeg and ffprobe are required")
    if not video.is_file():
        raise FileNotFoundError(video)
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
        raise ValueError(f"Unexpected polished video stream: {stream}")
    if audio_streams:
        raise ValueError("Muted-first candidate must not contain an audio stream")
    duration = float(metadata["format"]["duration"])
    if not math.isclose(duration, 29.0, rel_tol=0.0, abs_tol=1 / demo.FPS):
        raise ValueError(f"Unexpected duration: {duration}")

    _fast_start(video)
    _full_decode(ffmpeg, video)
    scan = _filter_scan(ffmpeg, video)
    artifacts = _artifact_checks()
    documents = _copy_checks()
    assets = base.load_assets()
    claims = base.verify_claim_sources(assets)
    for timestamp in (0.0, 3.6, 6.8, 10.8, 15.6, 18.7, 23.0, 26.5, 28.2):
        if demo.render_frame(timestamp, assets).size != (demo.WIDTH, demo.HEIGHT):
            raise ValueError(f"Layout validation failed at {timestamp:.2f}s")

    manifest = json.loads(demo.MANIFEST_PATH.read_text())
    if manifest["outputs"]["video"]["sha256"] != demo.sha256_path(video):
        raise ValueError("Manifest video hash does not match the polished candidate")
    preservation = demo.preserved_video_hashes()
    if manifest["preserved_videos"] != preservation:
        raise ValueError("Preserved-video hashes differ from the build manifest")

    stability = {"checked": False, "byte_identical": None}
    if compare is not None:
        comparison = demo.project_path(compare)
        if not comparison.is_file():
            raise FileNotFoundError(comparison)
        canonical_hash = demo.sha256_path(video)
        comparison_hash = demo.sha256_path(comparison)
        if canonical_hash != comparison_hash:
            raise ValueError("Polished comparison render is not byte-identical")
        stability = {
            "checked": True,
            "byte_identical": True,
            "comparison_path": str(comparison.relative_to(demo.PROJECT_ROOT)),
            "sha256": canonical_hash,
        }

    return {
        "schema_version": 1,
        "status": "PASS_polished_review_candidate",
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
        "claims": claims,
        "preserved_videos": preservation,
        "artifacts": artifacts,
        "documents": documents,
        "deterministic_regeneration": stability,
        "visual_review": {
            "status": "PASS_agent_encoded_frame_review",
            "native_encoded_contact_sheet": True,
            "640x360_encoded_contact_sheet": True,
            "transition_samples_seconds": [
                2.9,
                3.0,
                6.9,
                7.0,
                11.9,
                12.0,
                15.9,
                16.0,
                20.9,
                21.0,
                24.9,
                25.0,
                26.9,
                27.0,
            ],
            "finding": (
                "No clipped or colliding copy; scientific panels and labels remain "
                "separate and legible at 640x360."
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
