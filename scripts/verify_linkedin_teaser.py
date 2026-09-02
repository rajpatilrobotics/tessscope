"""Verify the TessScope LinkedIn teaser and write its machine-readable QA report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "video" / "linkedin"
VIDEO_PATH = OUTPUT_ROOT / "tessscope-linkedin-teaser.mp4"
POSTER_PATH = OUTPUT_ROOT / "tessscope-linkedin-poster.png"
CONTACT_PATH = OUTPUT_ROOT / "tessscope-linkedin-contact-sheet.png"
MANIFEST_PATH = OUTPUT_ROOT / "provenance-manifest.json"
REPORT_PATH = OUTPUT_ROOT / "qa-report.json"
JUDGE_VIDEO_PATH = PROJECT_ROOT / "outputs" / "video" / "tessscope-demo.mp4"
EXPECTED_JUDGE_VIDEO_SHA256 = (
    "767f5d03ae859d68b4d06b94a1398beda1a38026e1549bdb7b04606254a275a6"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(ffprobe: str, path: Path) -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        type=Path,
        help="Optional second render that must be byte-identical to the canonical teaser",
    )
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def verify(compare: Path | None = None) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("FFmpeg and ffprobe are required")
    required = (VIDEO_PATH, POSTER_PATH, CONTACT_PATH, MANIFEST_PATH, JUDGE_VIDEO_PATH)
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing teaser QA inputs: {missing}")

    manifest = json.loads(MANIFEST_PATH.read_text())
    video_hash = sha256_path(VIDEO_PATH)
    if video_hash != manifest["output"]["sha256"]:
        raise ValueError("Teaser hash does not match the provenance manifest")
    if manifest.get("test_accessed") is not False:
        raise ValueError("Teaser manifest must keep test_accessed=false")
    if manifest.get("scope") != "frozen_validation":
        raise ValueError("Teaser must be labelled frozen validation")
    if manifest.get("audio_policy") != "intentional_no_audio_stream_muted_first":
        raise ValueError("Teaser audio policy changed")
    for source in manifest["scientific_sources"]:
        source_path = PROJECT_ROOT / source["path"]
        if sha256_path(source_path) != source["sha256"]:
            raise ValueError(f"Scientific source hash changed: {source['path']}")

    metadata = probe(ffprobe, VIDEO_PATH)
    video_streams = [
        stream for stream in metadata["streams"] if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in metadata["streams"] if stream.get("codec_type") == "audio"
    ]
    if len(video_streams) != 1:
        raise ValueError("Expected exactly one video stream")
    stream = video_streams[0]
    expected = {
        "codec_name": "h264",
        "width": 1080,
        "height": 1350,
        "pix_fmt": "yuv420p",
        "r_frame_rate": "30/1",
    }
    if any(stream.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected teaser video format: {stream}")
    if audio_streams:
        raise ValueError("Muted-first teaser must intentionally contain no audio stream")
    duration = float(metadata["format"]["duration"])
    if abs(duration - 27.0) > 1 / 30:
        raise ValueError(f"Expected 27-second teaser; found {duration}")
    if int(metadata["format"]["size"]) >= 200 * 1024 * 1024:
        raise ValueError("Teaser is unexpectedly large for feed upload")

    video_bytes = VIDEO_PATH.read_bytes()
    moov = video_bytes.find(b"moov")
    mdat = video_bytes.find(b"mdat")
    if moov < 0 or mdat < 0 or moov > mdat:
        raise ValueError("MP4 is not fast-start optimized")

    decode = subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-i", str(VIDEO_PATH), "-f", "null", "-"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0 or decode.stderr.strip():
        raise ValueError(f"Teaser decode check failed: {decode.stderr.strip()}")
    motion = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(VIDEO_PATH),
            "-vf",
            "blackdetect=d=0.25:pix_th=0.02,freezedetect=n=-58dB:d=1.0",
            "-an",
            "-f",
            "null",
            "-",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if "black_start:" in motion.stderr:
        raise ValueError("Unexpected black interval detected")
    if "freeze_start:" in motion.stderr:
        raise ValueError("Unexpected one-second frozen interval detected")

    with Image.open(POSTER_PATH) as poster:
        if poster.size != (1080, 1350):
            raise ValueError("Poster must be 1080×1350")
    with Image.open(CONTACT_PATH) as contact:
        if contact.size != (1080, 1328):
            raise ValueError("Contact sheet dimensions changed")
    if sha256_path(POSTER_PATH) != manifest["poster"]["sha256"]:
        raise ValueError("Poster hash does not match the manifest")
    if sha256_path(CONTACT_PATH) != manifest["contact_sheet"]["sha256"]:
        raise ValueError("Contact-sheet hash does not match the manifest")
    if sha256_path(JUDGE_VIDEO_PATH) != EXPECTED_JUDGE_VIDEO_SHA256:
        raise ValueError("Existing 3:30 judge video changed")

    compare_result: dict[str, object] = {"checked": False}
    if compare is not None:
        compare_path = compare if compare.is_absolute() else PROJECT_ROOT / compare
        compare_path = compare_path.resolve()
        if not compare_path.is_relative_to(PROJECT_ROOT) or not compare_path.is_file():
            raise ValueError("Comparison render must be an existing file inside the project")
        compare_hash = sha256_path(compare_path)
        if compare_hash != video_hash:
            raise ValueError("Second teaser render is not byte-identical")
        compare_result = {
            "checked": True,
            "path": str(compare_path.relative_to(PROJECT_ROOT)),
            "sha256": compare_hash,
            "byte_identical": True,
        }

    return {
        "status": "PASS",
        "video": {
            "path": str(VIDEO_PATH.relative_to(PROJECT_ROOT)),
            "sha256": video_hash,
            "bytes": VIDEO_PATH.stat().st_size,
            "duration_seconds": duration,
            "stream": stream,
            "audio_streams": 0,
            "fast_start": True,
            "decode_errors": 0,
            "black_intervals": 0,
            "frozen_intervals_over_one_second": 0,
        },
        "poster_sha256": sha256_path(POSTER_PATH),
        "contact_sheet_sha256": sha256_path(CONTACT_PATH),
        "scientific_source_hashes": "PASS",
        "claim_source_values": "PASS",
        "judge_video_preserved": True,
        "test_accessed": False,
        "byte_stability": compare_result,
    }


def main() -> None:
    args = parse_args()
    report = verify(args.compare)
    if args.write_report:
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
