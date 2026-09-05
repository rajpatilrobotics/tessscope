"""Build the compact, source-traceable visual set used by the top-level README."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "demo" / "readme"

FINAL_VIDEO = (
    PROJECT_ROOT
    / "outputs/video/linkedin/tessscope-end-to-end-demo-polished-v23.mp4"
)
FRAME_TIMESTAMPS_SECONDS = {
    "video_poster": 2.0,
    "problem": 8.0,
    "solution": 15.0,
    "system_proof": 89.7,
}
ANIMATED_CLIPS = {
    "video_preview_gif": {
        "filename": "video-preview.gif",
        "start_seconds": 0.4,
        "duration_seconds": 3.2,
    },
    "problem_gif": {
        "filename": "problem.gif",
        "start_seconds": 4.4,
        "duration_seconds": 5.2,
    },
    "solution_gif": {
        "filename": "solution.gif",
        "start_seconds": 10.4,
        "duration_seconds": 6.2,
    },
    "graph_gif": {
        "filename": "graph.gif",
        "start_seconds": 69.4,
        "duration_seconds": 5.2,
    },
    "system_proof_gif": {
        "filename": "system-proof.gif",
        "start_seconds": 84.4,
        "duration_seconds": 5.2,
    },
}
GIF_SIZE = (960, 540)
GIF_FPS = 8

FONT_ROOT = Path(get_data_path()) / "fonts" / "ttf"
FONT_REGULAR = FONT_ROOT / "DejaVuSans.ttf"
FONT_BOLD = FONT_ROOT / "DejaVuSans-Bold.ttf"

WHITE = "#FFFFFF"
NAVY = "#071A33"
CYAN = "#009CC6"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#7D4DB2"
GREY = "#5F7185"
LIGHT = "#DCE8EF"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def _centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def _frame_at(seconds: float) -> Image.Image:
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(FINAL_VIDEO),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return Image.open(io.BytesIO(process.stdout)).convert("RGB")


def _scaled_copy(source: Image.Image, destination: Path, size: tuple[int, int]) -> None:
    image = source.copy().convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(destination, format="PNG", compress_level=9)


def _export_gif(destination: Path, *, start_seconds: float, duration_seconds: float) -> dict:
    """Export a looping, palette-optimized GIF from the approved final video."""
    filter_graph = (
        f"fps={GIF_FPS},scale={GIF_SIZE[0]}:{GIF_SIZE[1]}:flags=lanczos,"
        "split[s0][s1];"
        "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(FINAL_VIDEO),
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-an",
            "-vf",
            filter_graph,
            "-loop",
            "0",
            "-gifflags",
            "+transdiff",
            "-map_metadata",
            "-1",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    with Image.open(destination) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        frame_duration_ms = int(image.info.get("duration", 0))
        loop = int(image.info.get("loop", 0))
        size = list(image.size)
    return {
        "size": size,
        "frame_count": frame_count,
        "fps": GIF_FPS,
        "frame_duration_ms": frame_duration_ms,
        "duration_seconds": round(frame_count * frame_duration_ms / 1000, 3),
        "loop": loop,
        "source_start_seconds": start_seconds,
        "source_duration_seconds": duration_seconds,
        "bytes": destination.stat().st_size,
    }


def _paste_crop(
    canvas: Image.Image,
    source: Image.Image,
    crop: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
) -> None:
    image = source.convert("RGB").crop(crop)
    width = box[2] - box[0]
    height = box[3] - box[1]
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, (box[0], box[1]))


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str,
    width: int = 6,
    head: int = 16,
) -> None:
    draw.line((start, end), fill=fill, width=width)
    x, y = end
    if end[0] >= start[0]:
        polygon = [(x, y), (x - head, y - head // 2), (x - head, y + head // 2)]
    else:
        polygon = [(x, y), (x + head, y - head // 2), (x + head, y + head // 2)]
    draw.polygon(polygon, fill=fill)


def _autofocus_icon(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    draw.line((left + 18, center_y - 45, right - 18, center_y - 45), fill=CYAN, width=5)
    draw.line((left + 45, center_y + 42, right - 45, center_y + 42), fill=NAVY, width=6)
    draw.rounded_rectangle(
        (left + 55, center_y + 4, right - 55, center_y + 33),
        radius=8,
        fill=NAVY,
    )
    draw.line((center_x, center_y + 1, center_x, center_y - 35), fill=ORANGE, width=6)
    draw.polygon(
        [
            (center_x, center_y - 48),
            (center_x - 11, center_y - 30),
            (center_x + 11, center_y - 30),
        ],
        fill=ORANGE,
    )
    for offset in (-48, 48):
        draw.line(
            (center_x + offset, center_y + 33, center_x + offset, center_y + 42),
            fill=NAVY,
            width=5,
        )


def _build_gradient_path(
    destination: Path,
    *,
    hook: Image.Image,
    solution: Image.Image,
) -> dict:
    canvas = Image.new("RGB", (1800, 900), WHITE)
    draw = ImageDraw.Draw(canvas)

    _centered_text(
        draw,
        (900, 58),
        "One exact gradient across three native runtimes",
        font=_font(48, bold=True),
        fill=NAVY,
    )
    _centered_text(
        draw,
        (900, 112),
        "Forward correction above · reverse learning signal below",
        font=_font(24),
        fill=GREY,
    )

    draw.text((90, 164), "FORWARD CLOSED LOOP", font=_font(23, bold=True), fill=CYAN)
    centers = [170, 530, 890, 1250, 1610]
    labels = [
        ("LEARNED OPTICS", "JAX · Chromatix", PURPLE),
        ("FIRST EXPOSURE", "JAX · Chromatix", BLUE),
        ("AUTOFOCUS + STAGE", "NumPy · SciPy", GREEN),
        ("CORRECTED EXPOSURE", "JAX · Chromatix", BLUE),
        ("BIOLOGICAL LOSS", "PyTorch · InstanSeg", ORANGE),
    ]

    thumb_y = 245
    thumb_size = 190
    boxes = [
        (centers[0] - 95, thumb_y, centers[0] + 95, thumb_y + thumb_size),
        (centers[1] - 95, thumb_y, centers[1] + 95, thumb_y + thumb_size),
        (centers[2] - 95, thumb_y, centers[2] + 95, thumb_y + thumb_size),
        (centers[3] - 95, thumb_y, centers[3] + 95, thumb_y + thumb_size),
        (centers[4] - 95, thumb_y, centers[4] + 95, thumb_y + thumb_size),
    ]

    _paste_crop(canvas, hook, (790, 391, 1130, 731), boxes[0])
    _paste_crop(canvas, solution, (110, 395, 590, 875), boxes[1])
    _autofocus_icon(canvas, boxes[2])
    _paste_crop(canvas, solution, (1330, 395, 1810, 875), boxes[3])
    _paste_crop(canvas, hook, (1358, 342, 1838, 822), boxes[4])

    for index, (title, stack, color) in enumerate(labels):
        center = centers[index]
        _centered_text(draw, (center, 215), title, font=_font(21, bold=True), fill=color)
        _centered_text(draw, (center, 468), stack, font=_font(20), fill=GREY)
        if index < len(centers) - 1:
            _arrow(
                draw,
                (boxes[index][2] + 22, 340),
                (boxes[index + 1][0] - 22, 340),
                fill=CYAN,
                width=5,
                head=18,
            )

    cut_x = (centers[2] + centers[3]) // 2
    for y in range(370, 596, 22):
        draw.line((cut_x, y, cut_x, min(y + 12, 596)), fill=ORANGE, width=4)
    _centered_text(
        draw,
        (cut_x, 625),
        "STOPPED CONTROL CUTS ONLY THIS BACKWARD EDGE",
        font=_font(18, bold=True),
        fill=ORANGE,
    )
    _centered_text(
        draw,
        (cut_x, 653),
        "forward values remain identical",
        font=_font(18),
        fill=GREY,
    )

    draw.text((90, 598), "REVERSE EXACT GRADIENT", font=_font(23, bold=True), fill=BLUE)
    _arrow(draw, (1670, 705), (125, 705), fill=BLUE, width=6, head=20)
    reverse_labels = [
        (centers[0], "PUPIL GRADIENT", "parameter update"),
        (centers[1], "JAX VJP", "phase + image formation"),
        (centers[2], "ANALYTIC + IMPLICIT VJP", "spectral features + ridge solve"),
        (centers[3], "JAX VJP", "residual depth + second optics call"),
        (centers[4], "AUTOGRAD VJP", "frozen InstanSeg observer"),
    ]
    for center, title, detail in reverse_labels:
        draw.line((center, 688, center, 722), fill=LIGHT, width=5)
        _centered_text(draw, (center, 752), title, font=_font(19, bold=True), fill=NAVY)
        _centered_text(draw, (center, 782), detail, font=_font(16), fill=GREY)

    draw.line((90, 826, 1710, 826), fill=LIGHT, width=2)
    _centered_text(
        draw,
        (900, 858),
        (
            "Derivative check: 0.004802 median relative error · 0.999967 cosine · "
            "0.0 exact/stopped forward difference"
        ),
        font=_font(19),
        fill=GREY,
    )

    canvas.save(destination, format="PNG", compress_level=9)
    return {
        "size": list(canvas.size),
        "source_crops": {
            "learned_pupil": ["video_poster@2.000s", [790, 391, 1130, 731]],
            "first_exposure": ["solution@15.000s", [110, 395, 590, 875]],
            "corrected_exposure": ["solution@15.000s", [1330, 395, 1810, 875]],
            "task_observation": ["video_poster@2.000s", [1358, 342, 1838, 822]],
        },
    }


def build_assets() -> dict:
    """Build README images and return their provenance manifest."""
    if not FINAL_VIDEO.is_file():
        raise FileNotFoundError(FINAL_VIDEO)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "video_poster": OUTPUT_ROOT / "video-poster.png",
        "problem": OUTPUT_ROOT / "problem.png",
        "solution": OUTPUT_ROOT / "solution.png",
        "system_proof": OUTPUT_ROOT / "system-proof.png",
        "gradient_path": OUTPUT_ROOT / "gradient-path.png",
    }
    animated_paths = {
        name: OUTPUT_ROOT / spec["filename"]
        for name, spec in ANIMATED_CLIPS.items()
    }
    source_frames = {
        name: _frame_at(timestamp)
        for name, timestamp in FRAME_TIMESTAMPS_SECONDS.items()
    }
    for name in ("video_poster", "problem", "solution", "system_proof"):
        _scaled_copy(source_frames[name], output_paths[name], (1600, 900))
    gradient_details = _build_gradient_path(
        output_paths["gradient_path"],
        hook=source_frames["video_poster"],
        solution=source_frames["solution"],
    )
    animated_details = {
        name: _export_gif(
            path,
            start_seconds=spec["start_seconds"],
            duration_seconds=spec["duration_seconds"],
        )
        for name, spec in ANIMATED_CLIPS.items()
        for path in (animated_paths[name],)
    }

    manifest = {
        "schema_version": 2,
        "purpose": "README-only images and animations derived from the approved final video",
        "sources": {str(FINAL_VIDEO.relative_to(PROJECT_ROOT)): _sha256(FINAL_VIDEO)},
        "frame_timestamps_seconds": FRAME_TIMESTAMPS_SECONDS,
        "outputs": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
                "size": list(Image.open(path).size),
            }
            for name, path in output_paths.items()
        },
        "animations": {
            name: {
                "path": str(animated_paths[name].relative_to(PROJECT_ROOT)),
                "sha256": _sha256(animated_paths[name]),
                **animated_details[name],
            }
            for name in ANIMATED_CLIPS
        },
        "gradient_path": gradient_details,
        "scientific_integrity": {
            "generated_scientific_imagery": False,
            "source_pixels": "approved video frames built from frozen TessScope evidence",
            "animated_source": "approved V23 MP4 only; no new scientific frames synthesized",
            "vector_only_addition": "autofocus-stage schematic and explanatory connectors",
        },
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build_assets()
    print(json.dumps({"status": "ok", "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
