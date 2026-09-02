"""Build and verify the 29-second source-derived TessScope LinkedIn fast cut."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "outputs" / "video" / "tessscope-demo.mp4"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "video" / "linkedin"
DEFAULT_OUTPUT = OUTPUT_ROOT / "tessscope-linkedin-fastcut.mp4"
CONTACT_SHEET_PATH = OUTPUT_ROOT / "tessscope-linkedin-fastcut-contact-sheet.png"
REPORT_PATH = OUTPUT_ROOT / "tessscope-linkedin-fastcut-qa.json"
REVIEW_ROOT = PROJECT_ROOT / ".release-check" / "linkedin-fastcut-correspondence"

SOURCE_DURATION_SECONDS = 210
OUTPUT_DURATION_SECONDS = 29
FPS = 30
OUTPUT_FRAME_COUNT = OUTPUT_DURATION_SECONDS * FPS
SPEED_FACTOR = Fraction(SOURCE_DURATION_SECONDS, OUTPUT_DURATION_SECONDS)
SETPTS_FILTER = "setpts=PTS*29/210,fps=fps=30:round=near:start_time=0"
EXPECTED_SOURCE_SHA256 = (
    "767f5d03ae859d68b4d06b94a1398beda1a38026e1549bdb7b04606254a275a6"
)
SAMPLE_OUTPUT_FRAMES = (15, 120, 240, 360, 480, 600, 720, 825, 865)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: Path) -> Path:
    path = value if value.is_absolute() else PROJECT_ROOT / value
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Fast-cut output must stay inside the project directory")
    return path


def source_frame_for_output(output_frame: int) -> int:
    if not 0 <= output_frame < OUTPUT_FRAME_COUNT:
        raise ValueError("Output frame is outside the 29-second timeline")
    return round(output_frame * float(SPEED_FACTOR))


def mapped_source_seconds(output_frame: int) -> float:
    return output_frame / FPS * float(SPEED_FACTOR)


def render_command(ffmpeg: str, output: Path) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(SOURCE_PATH),
        "-map",
        "0:v:0",
        "-vf",
        SETPTS_FILTER,
        "-frames:v",
        str(OUTPUT_FRAME_COUNT),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-fps_mode",
        "cfr",
        "-threads",
        "1",
        "-x264-params",
        "keyint=60:min-keyint=60:scenecut=0",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-metadata",
        "title=TessScope — 29-second source-derived fast cut",
        "-metadata",
        "comment=Uniform 210-to-29-second temporal compression; no spatial redesign",
        str(output),
    ]


def probe(ffprobe: str, path: Path) -> dict:
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
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def extract_frame(ffmpeg: str, video: Path, frame_index: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            "-fps_mode",
            "vfr",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def frame_correspondence(ffmpeg: str, output: Path) -> list[dict]:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_frame in SAMPLE_OUTPUT_FRAMES:
        source_frame = source_frame_for_output(output_frame)
        output_png = REVIEW_ROOT / f"output-{output_frame:04d}.png"
        source_png = REVIEW_ROOT / f"source-{source_frame:04d}.png"
        extract_frame(ffmpeg, output, output_frame, output_png)
        extract_frame(ffmpeg, SOURCE_PATH, source_frame, source_png)
        with Image.open(output_png) as output_image, Image.open(source_png) as source_image:
            output_array = np.asarray(output_image.convert("RGB"), dtype=np.float32) / 255.0
            source_array = np.asarray(source_image.convert("RGB"), dtype=np.float32) / 255.0
        absolute = np.abs(output_array - source_array)
        correlation = float(np.corrcoef(output_array.ravel(), source_array.ravel())[0, 1])
        row = {
            "output_frame": output_frame,
            "output_seconds": output_frame / FPS,
            "formula_source_seconds": mapped_source_seconds(output_frame),
            "matched_source_frame": source_frame,
            "matched_source_seconds": source_frame / FPS,
            "mean_absolute_error": float(absolute.mean()),
            "p99_absolute_error": float(np.percentile(absolute, 99)),
            "pixel_correlation": correlation,
        }
        if row["mean_absolute_error"] > 0.02 or correlation < 0.995:
            raise ValueError(f"Frame mapping correspondence failed: {row}")
        rows.append(row)
    return rows


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("A system Arial font is required for contact-sheet labels")
    return ImageFont.truetype(path, size)


def create_contact_sheet(ffmpeg: str, output: Path, mappings: list[dict]) -> None:
    thumbnails = []
    for row in mappings:
        frame_path = REVIEW_ROOT / f"output-{row['output_frame']:04d}.png"
        with Image.open(frame_path) as frame:
            thumbnails.append(frame.convert("RGB").resize((600, 338), Image.Resampling.LANCZOS))

    sheet = Image.new("RGB", (1920, 1190), (7, 19, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (45, 25),
        "TessScope 29-second fast cut · complete source-derived timeline",
        font=_font(34),
        fill=(245, 249, 252),
    )
    for index, (thumbnail, row) in enumerate(zip(thumbnails, mappings, strict=True)):
        grid_row, column = divmod(index, 3)
        x = 30 + column * 630
        y = 82 + grid_row * 362
        sheet.paste(thumbnail, (x, y))
        draw.rectangle((x, y + 296, x + 600, y + 338), fill=(4, 13, 21))
        label = (
            f"OUT {row['output_seconds']:05.2f}s  →  "
            f"SOURCE {row['matched_source_seconds']:06.2f}s"
        )
        draw.text((x + 14, y + 317), label, font=_font(22), fill=(99, 218, 255), anchor="lm")
    CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT_SHEET_PATH, optimize=False)


def verify_video(ffmpeg: str, ffprobe: str, output: Path, compare: Path | None) -> dict:
    if sha256_path(SOURCE_PATH) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Original judge-video hash changed")
    metadata = probe(ffprobe, output)
    video_streams = [
        stream for stream in metadata["streams"] if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in metadata["streams"] if stream.get("codec_type") == "audio"
    ]
    if len(video_streams) != 1:
        raise ValueError("Fast cut must contain exactly one video stream")
    stream = video_streams[0]
    expected = {
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "r_frame_rate": "30/1",
        "nb_read_frames": str(OUTPUT_FRAME_COUNT),
    }
    if any(stream.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected output stream: {stream}")
    if audio_streams:
        raise ValueError("Fast cut must omit the intentionally silent source audio")
    duration = float(metadata["format"]["duration"])
    if not math.isclose(duration, OUTPUT_DURATION_SECONDS, rel_tol=0.0, abs_tol=1 / FPS):
        raise ValueError(f"Fast-cut duration changed: {duration}")

    payload = output.read_bytes()
    moov = payload.find(b"moov")
    mdat = payload.find(b"mdat")
    if moov < 0 or mdat < 0 or moov > mdat:
        raise ValueError("Fast cut is not fast-start optimized")
    decode = subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-i", str(output), "-f", "null", "-"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if decode.returncode or decode.stderr.strip():
        raise ValueError(f"Full decode failed: {decode.stderr.strip()}")
    black = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(output),
            "-vf",
            "blackdetect=d=0.20:pix_th=0.02",
            "-an",
            "-f",
            "null",
            "-",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if "black_start:" in black.stderr:
        raise ValueError("Unexpected black interval detected")

    output_hash = sha256_path(output)
    stability = {"checked": False}
    if compare is not None:
        compare = project_path(compare)
        if not compare.is_file() or sha256_path(compare) != output_hash:
            raise ValueError("Comparison render is not byte-identical")
        stability = {
            "checked": True,
            "byte_identical": True,
            "comparison_path": str(compare.relative_to(PROJECT_ROOT)),
            "sha256": output_hash,
        }
    return {
        "metadata": metadata,
        "video_stream": stream,
        "audio_stream_count": 0,
        "duration_seconds": duration,
        "frame_count": OUTPUT_FRAME_COUNT,
        "fast_start": True,
        "decode_errors": 0,
        "black_intervals": 0,
        "byte_stability": stability,
    }


def build_report(
    ffmpeg: str,
    ffprobe: str,
    output: Path,
    compare: Path | None,
) -> dict:
    validation = verify_video(ffmpeg, ffprobe, output, compare)
    mappings = frame_correspondence(ffmpeg, output)
    create_contact_sheet(ffmpeg, output, mappings)
    return {
        "schema_version": 1,
        "status": "PASS_selected_source_derived_fastcut",
        "selected_candidate": True,
        "rejected_custom_4x5_teaser_selected": False,
        "source": {
            "path": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(SOURCE_PATH),
            "duration_seconds": SOURCE_DURATION_SECONDS,
            "preserved": True,
        },
        "temporal_transform": {
            "only_change": "uniform_time_compression",
            "speed_factor_exact": "210/29",
            "speed_factor_decimal": float(SPEED_FACTOR),
            "setpts": "PTS*29/210",
            "mapping": "t_source = t_output × 210/29",
            "spatial_filters": [],
            "selective_omissions": False,
            "scene_reordering": False,
            "new_overlays": False,
        },
        "output": {
            "path": str(output.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(output),
            "bytes": output.stat().st_size,
            **validation,
        },
        "frame_correspondence": mappings,
        "contact_sheet": {
            "path": str(CONTACT_SHEET_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(CONTACT_SHEET_PATH),
            "bytes": CONTACT_SHEET_PATH.stat().st_size,
            "samples": len(mappings),
        },
        "external_actions_performed": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--compare", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise SystemExit("FFmpeg and ffprobe are required")
    output = project_path(args.output)
    if args.verify_only and output != DEFAULT_OUTPUT.resolve():
        raise ValueError("--verify-only validates the canonical fast cut")
    if not args.verify_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(render_command(ffmpeg, output), cwd=PROJECT_ROOT, check=True)
    if output == DEFAULT_OUTPUT.resolve():
        report = build_report(ffmpeg, ffprobe, output, args.compare)
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2))
    else:
        print(
            json.dumps(
                {
                    "output": str(output.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_path(output),
                    "ffprobe": probe(ffprobe, output),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
