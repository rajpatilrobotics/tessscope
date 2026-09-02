"""Build the 3.5-minute caption-led TessScope submission video from traced assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = PROJECT_ROOT / "outputs" / "video"
DEFAULT_OUTPUT = VIDEO_ROOT / "tessscope-demo.mp4"
CAPTIONS = VIDEO_ROOT / "captions.srt"
MANIFEST = VIDEO_ROOT / "video-manifest.json"
CAPTION_BUILD_ROOT = PROJECT_ROOT / ".release-check" / "video-captions"
WIDTH = 1920
HEIGHT = 1080
FPS = 30


@dataclass(frozen=True)
class Shot:
    path: str
    duration_seconds: int
    kind: str = "still"


@dataclass(frozen=True)
class Caption:
    start_seconds: float
    end_seconds: float
    text: str

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


SHOTS = (
    Shot("outputs/demo/figures/hero-evidence.png", 22),
    Shot("outputs/demo/figures/gradient-architecture.png", 32),
    Shot("outputs/demo/figures/pupil-psf-depth.png", 24),
    Shot("outputs/demo/replay/validation-depth-sweep.mp4", 28, "video"),
    Shot("outputs/demo/figures/matched-microscopy.png", 28),
    Shot("outputs/demo/figures/pq-focus-depth.png", 24),
    Shot("outputs/demo/figures/causal-comparison.png", 38),
    Shot("outputs/demo/figures/hero-evidence.png", 14),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_output(path: Path) -> Path:
    output = path if path.is_absolute() else PROJECT_ROOT / path
    output = output.resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Video output must stay inside the project directory")
    return output


def timestamp_seconds(value: str) -> float:
    hours, minutes, remainder = value.split(":")
    seconds, milliseconds = remainder.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


def parse_captions() -> list[Caption]:
    captions = []
    for block in CAPTIONS.read_text().strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError(f"Invalid SRT block: {block}")
        start, end = lines[1].split(" --> ")
        captions.append(
            Caption(timestamp_seconds(start), timestamp_seconds(end), " ".join(lines[2:]))
        )
    expected_start = 0.0
    for caption in captions:
        if abs(caption.start_seconds - expected_start) > 1e-6:
            raise ValueError("Captions must be contiguous from zero")
        if caption.duration_seconds <= 0:
            raise ValueError("Caption durations must be positive")
        expected_start = caption.end_seconds
    if abs(expected_start - sum(shot.duration_seconds for shot in SHOTS)) > 1e-6:
        raise ValueError("Captions must end at the video duration")
    return captions


def render_caption_cards(captions: list[Caption]) -> list[Path]:
    CAPTION_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    font_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
        fallback_to_default=True,
    )
    font = ImageFont.truetype(font_path, 42)
    paths = []
    for index, caption in enumerate(captions, start=1):
        output = CAPTION_BUILD_ROOT / f"caption-{index:02d}.png"
        canvas = Image.new("RGBA", (WIDTH, 180), (7, 19, 26, 238))
        draw = ImageDraw.Draw(canvas)
        lines = textwrap.wrap(caption.text, width=78, break_long_words=False)
        if len(lines) > 2:
            raise ValueError(f"Caption does not fit two lines: {caption.text}")
        line_height = 52
        top = (180 - len(lines) * line_height) // 2
        for line_index, line in enumerate(lines):
            box = draw.textbbox((0, 0), line, font=font)
            width = box[2] - box[0]
            draw.text(
                ((WIDTH - width) / 2, top + line_index * line_height),
                line,
                font=font,
                fill=(255, 255, 255, 255),
            )
        canvas.save(output, optimize=False)
        paths.append(output)
    return paths


def build_command(
    ffmpeg: str, output: Path, captions: list[Caption], caption_cards: list[Path]
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for shot in SHOTS:
        source = PROJECT_ROOT / shot.path
        if not source.is_file():
            raise FileNotFoundError(f"Missing video source: {shot.path}")
        if shot.kind == "still":
            command.extend(
                [
                    "-loop",
                    "1",
                    "-framerate",
                    str(FPS),
                    "-t",
                    str(shot.duration_seconds),
                    "-i",
                    str(source),
                ]
            )
        else:
            command.extend(
                [
                    "-stream_loop",
                    "-1",
                    "-t",
                    str(shot.duration_seconds),
                    "-i",
                    str(source),
                ]
            )
    for caption, card in zip(captions, caption_cards, strict=True):
        command.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(FPS),
                "-t",
                str(caption.duration_seconds),
                "-i",
                str(card),
            ]
        )
    duration = sum(shot.duration_seconds for shot in SHOTS)
    audio_index = len(SHOTS) + len(captions)
    command.extend(
        [
            "-f",
            "lavfi",
            "-t",
            str(duration),
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
        ]
    )
    filters = []
    for index, shot in enumerate(SHOTS):
        filters.append(
            f"[{index}:v]"
            f"scale={WIDTH}:900:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:20:color=0x07131A,"
            f"setsar=1,fps={FPS},trim=duration={shot.duration_seconds},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
    inputs = "".join(f"[v{index}]" for index in range(len(SHOTS)))
    filters.append(f"{inputs}concat=n={len(SHOTS)}:v=1:a=0[base]")
    for index, caption in enumerate(captions):
        input_index = len(SHOTS) + index
        filters.append(
            f"[{input_index}:v]format=rgba,setsar=1,fps={FPS},"
            f"trim=duration={caption.duration_seconds},setpts=PTS-STARTPTS[c{index}]"
        )
    caption_inputs = "".join(f"[c{index}]" for index in range(len(captions)))
    filters.append(f"{caption_inputs}concat=n={len(captions)}:v=1:a=0[captiontrack]")
    filters.append("[base][captiontrack]overlay=0:900:format=auto[video]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[video]",
            "-map",
            f"{audio_index}:a",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            "-metadata",
            "title=TessScope — exact cross-runtime gradients",
            "-metadata",
            "comment=Cached validation evidence; no live inference or locked-test access",
            "-shortest",
            str(output),
        ]
    )
    return command


def probe(ffprobe: str, output: Path) -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    args = parse_args()
    output = project_output(args.output)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise SystemExit("FFmpeg and ffprobe are required to build the submission video")
    output.parent.mkdir(parents=True, exist_ok=True)
    captions = parse_captions()
    caption_cards = render_caption_cards(captions)
    subprocess.run(
        build_command(ffmpeg, output, captions, caption_cards),
        cwd=PROJECT_ROOT,
        check=True,
    )
    metadata = probe(ffprobe, output)
    if output == DEFAULT_OUTPUT.resolve():
        manifest = {
            "schema_version": 1,
            "status": "complete_caption_led_real_asset_video",
            "scientific_media_policy": "existing_traced_tessscope_assets_only",
            "cached_replay_not_live_inference": True,
            "test_accessed": False,
            "no_music_or_voice": True,
            "duration_seconds": sum(shot.duration_seconds for shot in SHOTS),
            "sources": [
                {
                    "path": shot.path,
                    "duration_seconds": shot.duration_seconds,
                    "kind": shot.kind,
                    "sha256": sha256_path(PROJECT_ROOT / shot.path),
                }
                for shot in SHOTS
            ],
            "captions": {
                "path": str(CAPTIONS.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(CAPTIONS),
            },
            "output": {
                "path": str(output.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(output),
                "bytes": output.stat().st_size,
                "ffprobe": metadata,
            },
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(output),
                "ffprobe": metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
