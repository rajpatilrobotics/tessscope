"""Render the 27-second muted-first TessScope LinkedIn teaser."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib import colormaps, font_manager
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tessscope.demo.figures import overlay_boundaries

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "video" / "linkedin"
DEFAULT_OUTPUT = OUTPUT_ROOT / "tessscope-linkedin-teaser.mp4"
POSTER_PATH = OUTPUT_ROOT / "tessscope-linkedin-poster.png"
CONTACT_SHEET_PATH = OUTPUT_ROOT / "tessscope-linkedin-contact-sheet.png"
MANIFEST_PATH = OUTPUT_ROOT / "provenance-manifest.json"
REPLAY_PATH = PROJECT_ROOT / "artifacts" / "runs" / "demo" / "validation-replay.npz"
REPLAY_METADATA_PATH = (
    PROJECT_ROOT / "artifacts" / "runs" / "demo" / "validation-replay.json"
)
VALIDATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)
DERIVATIVE_PATH = (
    PROJECT_ROOT / "artifacts" / "runs" / "v2_3" / "gates" / "closed-loop-derivative.json"
)
VISUAL_CONTRACT_PATH = PROJECT_ROOT / "configs" / "demo" / "visual-contract.yaml"
STORYBOARD_PATH = OUTPUT_ROOT / "storyboard.md"
COPY_PATH = OUTPUT_ROOT / "on-screen-copy.md"
JUDGE_VIDEO_PATH = PROJECT_ROOT / "outputs" / "video" / "tessscope-demo.mp4"

WIDTH = 1080
HEIGHT = 1350
FPS = 30
DURATION_SECONDS = 27
FRAME_COUNT = FPS * DURATION_SECONDS
SAFE_MARGIN = 72
TRANSITION_SECONDS = 0.28
DEPTHS_UM = (-6, -4, -2, 0, 2, 4, 6)
SCENES = (
    ("hook", 0.0, 3.0),
    ("correction", 3.0, 8.0),
    ("tesseract", 8.0, 13.0),
    ("optics", 13.0, 18.0),
    ("stage_result", 18.0, 23.0),
    ("causal_result", 23.0, 26.0),
    ("lockup", 26.0, 27.0),
)

NAVY = (4, 10, 24)
NAVY_2 = (7, 18, 38)
WHITE = (244, 249, 255)
MUTED = (151, 171, 194)
CYAN = (53, 220, 255)
BLUE = (58, 123, 255)
MAGENTA = (221, 91, 255)
VIOLET = (126, 87, 255)
ORANGE = (255, 145, 64)
GREEN = (66, 222, 169)
PANEL = (12, 26, 49)

EXPECTED_JUDGE_VIDEO_SHA256 = (
    "767f5d03ae859d68b4d06b94a1398beda1a38026e1549bdb7b04606254a275a6"
)


@dataclass(frozen=True)
class Assets:
    """Pre-rendered scientific panels and frozen source metadata."""

    before: Image.Image
    corrected: Image.Image
    pupil: Image.Image
    psfs: tuple[Image.Image, ...]
    metadata: dict
    validation: dict
    font_regular_path: Path
    font_bold_path: Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def ease(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def progress(value: float, start: float, end: float) -> float:
    if end <= start:
        raise ValueError("Progress interval must be positive")
    return ease((value - start) / (end - start))


def _rgb(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    return Image.fromarray(np.round(clipped * 255).astype(np.uint8), mode="RGB")


def load_assets() -> Assets:
    metadata = _json(REPLAY_METADATA_PATH)
    validation = _json(VALIDATION_PATH)
    if sha256_path(REPLAY_PATH) != metadata["archive"]["sha256"]:
        raise ValueError("Replay archive hash no longer matches its frozen metadata")

    with np.load(REPLAY_PATH, allow_pickle=False) as replay:
        exact_index = metadata["system_order"].index("exact")
        corrected_index = metadata["corrected_system_order"].index("exact")
        depth_index = metadata["depths_um"].index(-6.0)
        scale = float(metadata["observer"]["transform"]["scale"])
        target = replay["target_labels"]
        before = overlay_boundaries(
            replay["sensor_before"][exact_index, depth_index],
            target,
            replay["labels_before"][exact_index, depth_index],
            scale=scale,
        )
        corrected = overlay_boundaries(
            replay["sensor_corrected"][corrected_index, depth_index],
            target,
            replay["labels_corrected"][corrected_index, depth_index],
            scale=scale,
        )

        mask = replay["pupil_mask"].astype(bool)
        phase = np.angle(
            np.exp(1j * replay["pupil_phase_radians"][exact_index])
        ).astype(np.float32)
        phase_rgba = np.zeros((*phase.shape, 4), dtype=np.uint8)
        phase_colors = colormaps["twilight"]((phase + np.pi) / (2 * np.pi))
        phase_rgba[..., :3] = np.round(phase_colors[..., :3] * 255).astype(np.uint8)
        phase_rgba[..., 3] = mask.astype(np.uint8) * 255

        psf_stack = replay["psf_sensor"][exact_index]
        global_maximum = float(replay["psf_sensor"].max())
        start = (psf_stack.shape[-1] - 41) // 2
        psfs = []
        for psf in psf_stack:
            cropped = psf[start : start + 41, start : start + 41]
            log_psf = np.log10(np.maximum(cropped / global_maximum, 1e-5))
            colors = colormaps["magma"]((log_psf + 5.0) / 5.0)[..., :3]
            psfs.append(_rgb(colors))

    regular = Path(
        font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans"), fallback_to_default=True
        )
    )
    bold = Path(
        font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
            fallback_to_default=True,
        )
    )
    return Assets(
        before=_rgb(before),
        corrected=_rgb(corrected),
        pupil=Image.fromarray(phase_rgba, mode="RGBA"),
        psfs=tuple(psfs),
        metadata=metadata,
        validation=validation,
        font_regular_path=regular,
        font_bold_path=bold,
    )


@lru_cache(maxsize=64)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _base_background() -> Image.Image:
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    vertical = yy / max(HEIGHT - 1, 1)
    horizontal = xx / max(WIDTH - 1, 1)
    top = np.asarray(NAVY_2, dtype=np.float32)
    bottom = np.asarray(NAVY, dtype=np.float32)
    pixels = top[None, None, :] * (1.0 - vertical[..., None])
    pixels += bottom[None, None, :] * vertical[..., None]
    pixels += 4.0 * np.sin(horizontal[..., None] * np.pi)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


BACKGROUND = _base_background()


@lru_cache(maxsize=32)
def _glow(color: tuple[int, int, int], diameter: int, alpha: int = 150) -> Image.Image:
    layer = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    pad = diameter // 4
    draw.ellipse((pad, pad, diameter - pad, diameter - pad), fill=(*color, alpha))
    return layer.filter(ImageFilter.GaussianBlur(diameter // 8))


GLOW_CYAN = _glow(CYAN, 620, 100)
GLOW_MAGENTA = _glow(MAGENTA, 700, 105)


def canvas(time_seconds: float, *, magenta: bool = False) -> Image.Image:
    image = BACKGROUND.copy().convert("RGBA")
    glow = GLOW_MAGENTA if magenta else GLOW_CYAN
    x = int(-230 + 35 * math.sin(time_seconds * 0.42))
    y = int(690 + 25 * math.cos(time_seconds * 0.37))
    image.alpha_composite(glow, (x, y))
    second_x = int(730 + 24 * math.cos(time_seconds * 0.31))
    image.alpha_composite(GLOW_MAGENTA if not magenta else GLOW_CYAN, (second_x, -250))
    draw = ImageDraw.Draw(image)
    for x_grid in range(0, WIDTH, 90):
        draw.line((x_grid, 0, x_grid, HEIGHT), fill=(72, 110, 145, 12), width=1)
    for index in range(18):
        x_dot = int((index * 173 + 90 + 16 * math.sin(time_seconds + index)) % WIDTH)
        y_dot = int((index * 241 + 130 - time_seconds * (5 + index % 3)) % HEIGHT)
        radius = 1 + index % 2
        draw.ellipse(
            (x_dot - radius, y_dot - radius, x_dot + radius, y_dot + radius),
            fill=(*CYAN, 28 + 5 * (index % 4)),
        )
    return image


def _font(assets: Assets, size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = assets.font_bold_path if bold else assets.font_regular_path
    return font(str(path), size)


def _center_text(
    image: Image.Image,
    y: int,
    text: str,
    assets: Assets,
    *,
    size: int,
    color: tuple[int, int, int] = WHITE,
    bold: bool = False,
    spacing: int = 8,
) -> None:
    ImageDraw.Draw(image).multiline_text(
        (WIDTH // 2, y),
        text,
        font=_font(assets, size, bold=bold),
        fill=(*color, 255),
        anchor="ma",
        align="center",
        spacing=spacing,
    )


def _left_text(
    image: Image.Image,
    x: int,
    y: int,
    text: str,
    assets: Assets,
    *,
    size: int,
    color: tuple[int, int, int] = WHITE,
    bold: bool = False,
    anchor: str = "la",
) -> None:
    ImageDraw.Draw(image).text(
        (x, y),
        text,
        font=_font(assets, size, bold=bold),
        fill=(*color, 255),
        anchor=anchor,
    )


def _kicker(image: Image.Image, text: str, assets: Assets, *, color: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    text_font = _font(assets, 29, bold=True)
    box = draw.textbbox((0, 0), text, font=text_font)
    width = box[2] - box[0]
    x0 = (WIDTH - width) // 2 - 24
    y0 = 64
    draw.rounded_rectangle(
        (x0, y0, x0 + width + 48, y0 + 58),
        radius=29,
        fill=(*PANEL, 255),
        outline=(*color, 220),
        width=2,
    )
    draw.text(
        (WIDTH // 2, y0 + 29),
        text,
        font=text_font,
        fill=(*color, 255),
        anchor="mm",
    )


def _panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    border: tuple[int, int, int] = CYAN,
    alpha: int = 225,
    radius: int = 30,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=(*PANEL, alpha),
        outline=(*border, 140),
        width=2,
    )


@lru_cache(maxsize=16)
def _scan_texture(
    width: int,
    height: int,
    color: tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    texture = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(texture).rectangle(
        (width // 3, 0, 2 * width // 3, height),
        fill=(*color, alpha),
    )
    return texture.filter(ImageFilter.GaussianBlur(width // 5))


def _moving_scan(
    image: Image.Image,
    box: tuple[int, int, int, int],
    fraction: float,
    *,
    color: tuple[int, int, int],
    alpha: int = 48,
    width: int = 150,
) -> None:
    x0, y0, x1, y1 = box
    scan = _scan_texture(width, y1 - y0, color, alpha)
    x = int(x0 - width + (x1 - x0 + width) * (fraction % 1.0))
    image.alpha_composite(scan, (x, y0))


def _scientific_square(
    image: Image.Image,
    source: Image.Image,
    *,
    x: int,
    y: int,
    size: int,
    border: tuple[int, int, int] = CYAN,
    opacity: int = 255,
) -> None:
    glow = _glow(border, size + 80, 90)
    image.alpha_composite(glow, (x - 40, y - 40))
    resized = source.resize((size, size), Image.Resampling.NEAREST).convert("RGBA")
    if opacity != 255:
        resized.putalpha(opacity)
    image.alpha_composite(resized, (x, y))
    ImageDraw.Draw(image).rounded_rectangle(
        (x - 2, y - 2, x + size + 1, y + size + 1),
        radius=8,
        outline=(*border, 210),
        width=3,
    )


def _moving_science_background(
    image: Image.Image,
    source: Image.Image,
    local: float,
    *,
    darkness: int = 125,
) -> None:
    zoom = 930 + int(50 * ease(local / 3.0))
    scientific = source.resize((zoom, zoom), Image.Resampling.NEAREST).convert("RGBA")
    x = WIDTH - zoom + int(18 * math.sin(local * 0.8))
    y = 245 + int(15 * math.cos(local * 0.6))
    image.alpha_composite(scientific, (x, y))
    veil = Image.new("RGBA", (WIDTH, HEIGHT), (2, 8, 20, darkness))
    image.alpha_composite(veil)
    shade = Image.new("RGBA", (WIDTH, 520), (2, 8, 20, 0))
    shade_pixels = np.zeros((520, WIDTH, 4), dtype=np.uint8)
    shade_pixels[..., :3] = np.asarray(NAVY, dtype=np.uint8)
    shade_pixels[..., 3] = np.linspace(245, 0, 520, dtype=np.uint8)[:, None]
    shade = Image.fromarray(shade_pixels, mode="RGBA")
    image.alpha_composite(shade, (0, 0))


def scene_hook(local: float, assets: Assets) -> Image.Image:
    image = canvas(local, magenta=True)
    _moving_science_background(image, assets.before, local, darkness=95)
    psf_size = 300
    psf = assets.psfs[0].resize((psf_size, psf_size), Image.Resampling.NEAREST).convert("RGBA")
    psf.putalpha(225)
    image.alpha_composite(_glow(MAGENTA, 390, 160), (665, 875))
    image.alpha_composite(psf, (710, 920))
    _kicker(image, "REAL VALIDATION OPTICS · −6 µm", assets, color=MAGENTA)
    _center_text(image, 176, "WHAT IF A MICROSCOPE", assets, size=48, bold=True)
    _center_text(image, 250, "COULD LEARN TO", assets, size=66, bold=True)
    _center_text(image, 335, "REFOCUS ITSELF?", assets, size=76, color=CYAN, bold=True)
    _left_text(image, 734, 1230, "TRACED PSF", assets, size=24, color=MUTED, bold=True)
    return image


def scene_correction(local: float, assets: Assets) -> Image.Image:
    image = canvas(local)
    step = 1 if local < 1.55 else 2 if local < 3.05 else 3
    labels = {
        1: "1 · SEE DEFOCUS",
        2: "2 · PREDICT A STAGE MOVE",
        3: "3 · FORM A CORRECTED EXPOSURE",
    }
    _kicker(image, labels[step], assets, color=CYAN if step != 2 else MAGENTA)

    size = 872
    x = (WIDTH - size) // 2
    y = 205
    before = assets.before.resize((size, size), Image.Resampling.NEAREST).convert("RGBA")
    after = assets.corrected.resize((size, size), Image.Resampling.NEAREST).convert("RGBA")
    image.alpha_composite(before, (x, y))
    wipe = progress(local, 1.75, 3.35)
    wipe_x = int(size * wipe)
    if wipe_x > 0:
        after_crop = after.crop((0, 0, wipe_x, size))
        image.alpha_composite(after_crop, (x, y))
        bar_x = x + wipe_x
        ImageDraw.Draw(image).line((bar_x, y, bar_x, y + size), fill=(*CYAN, 255), width=8)
        image.alpha_composite(_glow(CYAN, 180, 130), (bar_x - 90, y + size // 2 - 90))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (x - 3, y - 3, x + size + 3, y + size + 3),
        radius=12,
        outline=(*CYAN, 200),
        width=3,
    )
    _moving_scan(image, (x, y, x + size, y + size), local / 1.45, color=CYAN, alpha=36)
    label = "BEFORE" if wipe < 0.5 else "CORRECTED"
    label_color = MUTED if wipe < 0.5 else CYAN
    _left_text(image, x + 28, y + 28, label, assets, size=29, color=label_color, bold=True)

    action_alpha = int(255 * progress(local, 1.25, 1.8) * (1.0 - progress(local, 3.2, 3.9)))
    if action_alpha > 0:
        center = WIDTH // 2
        draw.line((center - 190, 1134, center + 190, 1134), fill=(*MAGENTA, action_alpha), width=8)
        draw.polygon(
            ((center + 190, 1134), (center + 154, 1112), (center + 154, 1156)),
            fill=(*MAGENTA, action_alpha),
        )
        _center_text(
            image,
            1088,
            "PREDICTED STAGE ACTION",
            assets,
            size=25,
            color=MAGENTA,
            bold=True,
        )
    _center_text(
        image,
        1260,
        "TRACED VALIDATION FIELD · IDENTICAL DISPLAY TRANSFORM",
        assets,
        size=23,
        color=MUTED,
        bold=True,
    )
    return image


def scene_tesseract(local: float, assets: Assets) -> Image.Image:
    image = canvas(local, magenta=True)
    _kicker(image, "TESSERACT BRIDGES 3 RUNTIMES", assets, color=CYAN)
    _center_text(image, 140, "ONE EXACT GRADIENT\nCROSSES THEM ALL", assets, size=55, bold=True)
    boxes = (
        (260, "OPTICS TESSERACT API", "JAX · CHROMATIX", CYAN),
        (590, "AUTOFOCUS TESSERACT API", "NUMPY · SCIPY", GREEN),
        (920, "OBSERVER TESSERACT API", "PYTORCH · INSTANSEG", ORANGE),
    )
    draw = ImageDraw.Draw(image)
    for y, title, runtime, color in boxes:
        _panel(image, (112, y, 968, y + 214), border=color, radius=32)
        _left_text(image, 154, y + 42, title, assets, size=27, color=color, bold=True)
        _left_text(image, 154, y + 111, runtime, assets, size=45, bold=True)
        draw.line((820, y + 44, 820, y + 170), fill=(*color, 95), width=2)
        _left_text(image, 858, y + 107, "API", assets, size=25, color=MUTED, bold=True, anchor="mm")
    for y0, y1 in ((474, 590), (804, 920)):
        draw.line((540, y0, 540, y1), fill=(*MUTED, 100), width=4)
        draw.polygon(((540, y1), (526, y1 - 20), (554, y1 - 20)), fill=(*MUTED, 150))

    _moving_scan(
        image,
        (112, 260, 968, 1134),
        local / 1.7,
        color=MAGENTA,
        alpha=30,
        width=180,
    )

    forward_p = progress(local, 0.65, 2.15)
    fy = int(298 + forward_p * 830)
    draw.line((910, 292, 910, 1135), fill=(*CYAN, 80), width=4)
    image.alpha_composite(_glow(CYAN, 130, 170), (845, fy - 65))
    draw.ellipse((895, fy - 15, 925, fy + 15), fill=(*CYAN, 255))

    reverse_p = progress(local, 2.25, 4.45)
    ry = int(1128 - reverse_p * 830)
    draw.line((850, 1135, 850, 292), fill=(*MAGENTA, 100), width=5)
    image.alpha_composite(_glow(MAGENTA, 145, 190), (778, ry - 72))
    draw.ellipse((831, ry - 19, 869, ry + 19), fill=(*MAGENTA, 255))
    _center_text(
        image,
        1210,
        "FORWARD SIGNAL ↓   ·   EXACT GRADIENT ↑",
        assets,
        size=28,
        color=CYAN if local < 2.3 else MAGENTA,
        bold=True,
    )
    return image


def scene_optics(local: float, assets: Assets) -> Image.Image:
    image = canvas(local)
    _kicker(image, "PHASE → PSF → SECOND EXPOSURE", assets, color=MAGENTA)
    _center_text(
        image,
        145,
        "A LEARNED PHASE PUPIL\nENCODES SIGNED DEFOCUS",
        assets,
        size=49,
        bold=True,
    )
    phase_size = 398
    phase_x, phase_y = 82, 358
    image.alpha_composite(_glow(VIOLET, 480, 145), (phase_x - 40, phase_y - 40))
    pupil = assets.pupil.resize((phase_size, phase_size), Image.Resampling.NEAREST)
    image.alpha_composite(pupil, (phase_x, phase_y))
    _left_text(
        image,
        phase_x + phase_size // 2,
        775,
        "EXACT B7 PUPIL",
        assets,
        size=25,
        color=MUTED,
        bold=True,
        anchor="ma",
    )

    psf_index = min(6, max(0, int(local / 0.55)))
    psf_size = 398
    psf_x, psf_y = 602, 358
    image.alpha_composite(_glow(MAGENTA, 480, 150), (psf_x - 40, psf_y - 40))
    psf = assets.psfs[psf_index].resize((psf_size, psf_size), Image.Resampling.NEAREST).convert(
        "RGBA"
    )
    image.alpha_composite(psf, (psf_x, psf_y))
    depth = DEPTHS_UM[psf_index]
    depth_text = f"{depth:+d} µm" if depth else "0 µm"
    _left_text(
        image,
        psf_x + psf_size // 2,
        775,
        f"CHROMATIX PSF · {depth_text}",
        assets,
        size=23,
        color=MUTED,
        bold=True,
        anchor="ma",
    )

    draw = ImageDraw.Draw(image)
    draw.line((486, 560, 590, 560), fill=(*CYAN, 220), width=7)
    draw.polygon(((590, 560), (562, 543), (562, 577)), fill=(*CYAN, 240))

    reveal = progress(local, 2.15, 3.25)
    second_size = int(470 * reveal)
    if second_size > 20:
        second_x = (WIDTH - second_size) // 2
        second_y = 842 + (470 - second_size) // 2
        _scientific_square(
            image,
            assets.corrected,
            x=second_x,
            y=second_y,
            size=second_size,
            border=CYAN,
        )
        _center_text(image, 810, "TRACED SECOND EXPOSURE", assets, size=29, color=CYAN, bold=True)
    _moving_scan(
        image,
        (82, 358, 1000, 1312),
        local / 1.6,
        color=CYAN,
        alpha=32,
        width=170,
    )
    return image


def scene_stage_result(local: float, assets: Assets) -> Image.Image:
    image = canvas(local)
    _scientific_square(image, assets.corrected, x=78, y=245, size=924, border=CYAN)
    veil = Image.new("RGBA", (WIDTH, HEIGHT), (2, 8, 20, 76))
    image.alpha_composite(veil)
    _kicker(image, "STAGE CORRECTION · FROZEN VALIDATION", assets, color=CYAN)

    slide = 110 * (1.0 - progress(local, 0.45, 1.25))
    top = int(865 + slide)
    _panel(image, (56, top, 1024, top + 410), border=CYAN, alpha=238, radius=36)
    _left_text(image, 98, top + 44, "PQ", assets, size=33, color=MUTED, bold=True)
    _left_text(image, 98, top + 100, "0.4922", assets, size=78, bold=True)
    _left_text(image, 430, top + 120, "→", assets, size=70, color=MAGENTA, bold=True)
    _left_text(image, 555, top + 92, "0.5523", assets, size=94, color=CYAN, bold=True)
    _left_text(image, 98, top + 230, "74.7%", assets, size=82, color=WHITE, bold=True)
    _left_text(
        image,
        432,
        top + 242,
        "OF HARD FRAMES\nIMPROVE",
        assets,
        size=36,
        color=CYAN,
        bold=True,
    )
    _left_text(
        image,
        98,
        top + 354,
        "27 HARD-DENSITY WELLS  ·  TRACED FIELD SHOWN",
        assets,
        size=21,
        color=MUTED,
        bold=True,
    )
    _moving_scan(
        image,
        (78, 245, 1002, 1275),
        local / 1.8,
        color=CYAN,
        alpha=34,
        width=170,
    )
    return image


def scene_causal_result(local: float, assets: Assets) -> Image.Image:
    image = canvas(local, magenta=True)
    _kicker(image, "CAUSAL PROOF · FROZEN VALIDATION", assets, color=MAGENTA)
    _center_text(
        image,
        156,
        "EXACT FEEDBACK GRADIENTS\nBEAT STOPPED GRADIENTS",
        assets,
        size=50,
        bold=True,
    )
    _center_text(image, 405, "+0.0063 PQ", assets, size=112, color=CYAN, bold=True)
    _center_text(image, 548, "FORWARD-IDENTICAL CONTROL", assets, size=25, color=MUTED, bold=True)

    minimum, maximum = -0.002, 0.013
    x0, x1 = 132, 948

    def scale(value: float) -> int:
        return int(x0 + (value - minimum) / (maximum - minimum) * (x1 - x0))

    draw = ImageDraw.Draw(image)
    zero_x = scale(0.0)
    draw.line((x0, 790, x1, 790), fill=(*MUTED, 100), width=4)
    draw.line((zero_x, 694, zero_x, 886), fill=(*WHITE, 170), width=4)
    _left_text(image, zero_x, 914, "0", assets, size=24, color=MUTED, bold=True, anchor="ma")
    mean_x = scale(0.006258840610359453)
    low_x = scale(0.0006703181908061804)
    high_x = scale(0.011547458540670894)
    line_p = progress(local, 0.55, 1.65)
    shown_low = int(mean_x + (low_x - mean_x) * line_p)
    shown_high = int(mean_x + (high_x - mean_x) * line_p)
    draw.line((shown_low, 790, shown_high, 790), fill=(*CYAN, 255), width=12)
    draw.line((shown_low, 756, shown_low, 824), fill=(*CYAN, 255), width=7)
    draw.line((shown_high, 756, shown_high, 824), fill=(*CYAN, 255), width=7)
    image.alpha_composite(_glow(CYAN, 170, 170), (mean_x - 85, 705))
    draw.ellipse((mean_x - 22, 768, mean_x + 22, 812), fill=(*CYAN, 255))
    _center_text(
        image,
        960,
        "95% CI [+0.0007, +0.0115]",
        assets,
        size=40,
        color=WHITE,
        bold=True,
    )
    _center_text(image, 1035, "ABOVE ZERO", assets, size=31, color=GREEN, bold=True)
    _center_text(
        image,
        1210,
        "27 HARD-DENSITY WELLS · 2,000 PAIRED BOOTSTRAPS",
        assets,
        size=23,
        color=MUTED,
        bold=True,
    )
    return image


def scene_lockup(local: float, assets: Assets) -> Image.Image:
    image = canvas(local, magenta=True)
    _moving_science_background(image, assets.before, local, darkness=170)
    image.alpha_composite(_glow(CYAN, 540, 150), (270, 310))
    _center_text(image, 425, "TessScope", assets, size=124, color=WHITE, bold=True)
    _center_text(
        image,
        585,
        "DIFFERENTIABLE\nCLOSED-LOOP MICROSCOPY",
        assets,
        size=42,
        color=CYAN,
        bold=True,
    )
    _center_text(
        image,
        775,
        "TESSERACT HACKATHON 2026 · TRACK 05",
        assets,
        size=26,
        color=MUTED,
        bold=True,
    )
    return image


SCENE_RENDERERS = (
    scene_hook,
    scene_correction,
    scene_tesseract,
    scene_optics,
    scene_stage_result,
    scene_causal_result,
    scene_lockup,
)


def render_frame(time_seconds: float, assets: Assets) -> Image.Image:
    if not 0.0 <= time_seconds < DURATION_SECONDS:
        raise ValueError("Frame time must fall inside the teaser duration")
    index = next(
        index
        for index, (_, start, end) in enumerate(SCENES)
        if start <= time_seconds < end
    )
    _, start, end = SCENES[index]
    current = SCENE_RENDERERS[index](time_seconds - start, assets).convert("RGB")
    if index < len(SCENES) - 1 and time_seconds > end - TRANSITION_SECONDS:
        blend = progress(time_seconds, end - TRANSITION_SECONDS, end)
        following = SCENE_RENDERERS[index + 1](0.0, assets).convert("RGB")
        transition = canvas(time_seconds).convert("RGB")
        if blend < 0.5:
            return Image.blend(current, transition, ease(blend * 2.0))
        return Image.blend(transition, following, ease((blend - 0.5) * 2.0))
    if index == len(SCENES) - 1 and time_seconds > end - 0.16:
        blend = progress(time_seconds, end - 0.16, end)
        transition = canvas(time_seconds).convert("RGB")
        if blend < 0.5:
            return Image.blend(current, transition, ease(blend * 2.0))
        return Image.blend(
            transition,
            scene_hook(0.0, assets).convert("RGB"),
            ease((blend - 0.5) * 2.0),
        )
    return current


def render_poster(assets: Assets) -> Image.Image:
    image = canvas(20.5)
    _scientific_square(image, assets.corrected, x=62, y=250, size=956, border=CYAN)
    veil = Image.new("RGBA", (WIDTH, HEIGHT), (2, 8, 20, 98))
    image.alpha_composite(veil)
    _kicker(image, "TESSERACT HACKATHON 2026 · TRACK 05", assets, color=CYAN)
    _center_text(image, 150, "TessScope", assets, size=96, bold=True)
    _panel(image, (52, 792, 1028, 1285), border=CYAN, alpha=238, radius=38)
    _center_text(
        image,
        838,
        "A MICROSCOPE THAT LEARNS\nHOW TO REFOCUS ITSELF",
        assets,
        size=47,
        bold=True,
    )
    _center_text(image, 1010, "PQ 0.4922 → 0.5523", assets, size=73, color=CYAN, bold=True)
    _center_text(image, 1118, "74.7% OF HARD FRAMES IMPROVE", assets, size=33, bold=True)
    _center_text(image, 1198, "FROZEN VALIDATION", assets, size=23, color=MUTED, bold=True)
    return image.convert("RGB")


def output_path(value: Path) -> Path:
    output = value if value.is_absolute() else PROJECT_ROOT / value
    output = output.resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Teaser output must stay inside the project directory")
    return output


def ffmpeg_command(ffmpeg: str, output: Path) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-threads",
        "1",
        "-x264-params",
        "keyint=60:min-keyint=60:scenecut=0",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-metadata",
        "title=TessScope — differentiable closed-loop microscopy",
        "-metadata",
        "comment=Frozen validation teaser; intentional no-audio stream",
        str(output),
    ]


def encode_video(ffmpeg: str, output: Path, assets: Assets) -> None:
    process = subprocess.Popen(
        ffmpeg_command(ffmpeg, output),
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("FFmpeg input pipe was not created")
    try:
        for frame_index in range(FRAME_COUNT):
            frame = render_frame(frame_index / FPS, assets)
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    except BrokenPipeError as error:
        raise RuntimeError("FFmpeg stopped while receiving rendered frames") from error
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)


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
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def create_contact_sheet(ffmpeg: str, output: Path, assets: Assets) -> list[float]:
    timestamps = [0.5, 3.6, 6.6, 9.7, 14.5, 17.0, 20.5, 24.4, 26.45]
    review_root = PROJECT_ROOT / ".release-check" / "linkedin-teaser-contact"
    review_root.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    for index, timestamp in enumerate(timestamps):
        path = review_root / f"frame-{index + 1:02d}.png"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(output),
                "-ss",
                f"{timestamp:.3f}",
                "-frames:v",
                "1",
                "-vf",
                "scale=320:400:flags=lanczos",
                str(path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        thumbnails.append(Image.open(path).convert("RGB"))

    sheet = Image.new("RGB", (1080, 1328), NAVY)
    draw = ImageDraw.Draw(sheet)
    title_font = _font(assets, 30, bold=True)
    label_font = _font(assets, 20, bold=True)
    draw.text((40, 22), "TessScope LinkedIn teaser · encoded-frame QA", font=title_font, fill=WHITE)
    for index, (timestamp, thumbnail) in enumerate(zip(timestamps, thumbnails, strict=True)):
        row, column = divmod(index, 3)
        x = 30 + column * 350
        y = 78 + row * 414
        sheet.paste(thumbnail, (x, y))
        draw.rectangle((x, y + 356, x + 320, y + 400), fill=(3, 9, 20))
        draw.text(
            (x + 14, y + 378),
            f"{timestamp:05.2f}s",
            font=label_font,
            fill=CYAN,
            anchor="lm",
        )
    sheet.save(CONTACT_SHEET_PATH, optimize=False)
    return timestamps


def verify_claim_sources(assets: Assets) -> dict:
    name = "v2_3-exact-balanced-segmentation_only-step-14"
    stage = assets.validation["stage_correction"][name]
    causal = assets.validation["paired_evidence"][name]["corrected_vs_stopped"]
    expected = {
        "before_pq": 0.4921920085941332,
        "after_pq": 0.552294837627718,
        "improved_fraction": 0.7469135802469136,
        "causal_gain": 0.006258840610359453,
        "causal_ci_lower_95": 0.0006703181908061804,
        "causal_ci_upper_95": 0.011547458540670894,
    }
    actual = {
        "before_pq": stage["hard_off_focus_pq_before"],
        "after_pq": stage["hard_off_focus_pq_after"],
        "improved_fraction": stage["fraction_hard_frames_improved"],
        "causal_gain": causal["mean_difference"],
        "causal_ci_lower_95": causal["ci_lower_95"],
        "causal_ci_upper_95": causal["ci_upper_95"],
    }
    for key, value in expected.items():
        if not math.isclose(actual[key], value, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Frozen teaser claim changed: {key}")
    return actual


def write_manifest(
    output: Path,
    assets: Assets,
    metadata: dict,
    contact_timestamps: list[float],
) -> None:
    claims = verify_claim_sources(assets)
    manifest = {
        "schema_version": 1,
        "status": "complete_muted_first_linkedin_teaser",
        "scientific_media_policy": "existing_traced_tessscope_assets_only",
        "decorative_layers": [
            "deep-navy gradient",
            "faint grid and deterministic particles",
            "cyan/magenta glow",
            "service-boundary and signal-pulse motion",
            "deterministic typography",
        ],
        "scope": "frozen_validation",
        "test_accessed": False,
        "live_inference": False,
        "physical_microscope_validation_claimed": False,
        "audio_policy": "intentional_no_audio_stream_muted_first",
        "dimensions": {"width": WIDTH, "height": HEIGHT, "aspect_ratio": "4:5"},
        "fps": FPS,
        "duration_seconds": DURATION_SECONDS,
        "safe_margin_px": SAFE_MARGIN,
        "scene_timeline": [
            {"id": name, "start_seconds": start, "end_seconds": end}
            for name, start, end in SCENES
        ],
        "claim_sources": {
            "stage_correction": {
                "display": {
                    "before": "PQ 0.4922",
                    "after": "PQ 0.5523",
                    "improved": "74.7% of hard frames improve",
                },
                "raw": {
                    "before_pq": claims["before_pq"],
                    "after_pq": claims["after_pq"],
                    "improved_fraction": claims["improved_fraction"],
                },
                "path": str(VALIDATION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(VALIDATION_PATH),
                "pointers": [
                    "/stage_correction/v2_3-exact-balanced-segmentation_only-step-14/hard_off_focus_pq_before",
                    "/stage_correction/v2_3-exact-balanced-segmentation_only-step-14/hard_off_focus_pq_after",
                    "/stage_correction/v2_3-exact-balanced-segmentation_only-step-14/fraction_hard_frames_improved",
                ],
            },
            "exact_vs_stopped": {
                "display": {
                    "gain": "+0.0063 PQ",
                    "interval": "95% CI [+0.0007, +0.0115]",
                },
                "raw": {
                    "gain": claims["causal_gain"],
                    "ci_lower_95": claims["causal_ci_lower_95"],
                    "ci_upper_95": claims["causal_ci_upper_95"],
                },
                "path": str(VALIDATION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(VALIDATION_PATH),
                "pointers": [
                    "/paired_evidence/v2_3-exact-balanced-segmentation_only-step-14/corrected_vs_stopped"
                ],
            },
        },
        "scientific_sources": [
            {
                "path": str(REPLAY_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(REPLAY_PATH),
                "usage": [
                    "exact -6 µm first sensor and InstanSeg/reference boundaries",
                    "exact -6 µm corrected sensor and InstanSeg/reference boundaries",
                    "exact B7 wrapped phase pupil",
                    "exact fixed-global-scale PSFs across seven depths",
                ],
                "display_transform": assets.metadata["observer"]["transform"],
            },
            {
                "path": str(REPLAY_METADATA_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(REPLAY_METADATA_PATH),
                "usage": "array order, selected field, source pointers, and display transform",
            },
            {
                "path": str(DERIVATIVE_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(DERIVATIVE_PATH),
                "usage": "validated three-runtime exact-gradient architecture",
            },
            {
                "path": str(VISUAL_CONTRACT_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(VISUAL_CONTRACT_PATH),
                "usage": "fixed geometry, transform, palette, and evidence boundary",
            },
        ],
        "copy": {
            "path": str(COPY_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(COPY_PATH),
        },
        "storyboard": {
            "path": str(STORYBOARD_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(STORYBOARD_PATH),
        },
        "font": {
            "regular_file": assets.font_regular_path.name,
            "regular_sha256": sha256_path(assets.font_regular_path),
            "bold_file": assets.font_bold_path.name,
            "bold_sha256": sha256_path(assets.font_bold_path),
        },
        "preserved_judge_video": {
            "path": str(JUDGE_VIDEO_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(JUDGE_VIDEO_PATH),
        },
        "output": {
            "path": str(output.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(output),
            "bytes": output.stat().st_size,
            "ffprobe": metadata,
        },
        "poster": {
            "path": str(POSTER_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(POSTER_PATH),
            "bytes": POSTER_PATH.stat().st_size,
        },
        "contact_sheet": {
            "path": str(CONTACT_SHEET_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(CONTACT_SHEET_PATH),
            "bytes": CONTACT_SHEET_PATH.stat().st_size,
            "encoded_frame_timestamps_seconds": contact_timestamps,
            "thumbnail_dimensions": [320, 400],
        },
    }
    if manifest["preserved_judge_video"]["sha256"] != EXPECTED_JUDGE_VIDEO_SHA256:
        raise ValueError("The existing judge video changed during teaser production")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = output_path(args.output)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise SystemExit("FFmpeg and ffprobe are required to build the LinkedIn teaser")
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = load_assets()
    verify_claim_sources(assets)
    encode_video(ffmpeg, output, assets)
    metadata = probe(ffprobe, output)
    if output == DEFAULT_OUTPUT.resolve():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        render_poster(assets).save(POSTER_PATH, optimize=False)
        contact_timestamps = create_contact_sheet(ffmpeg, output, assets)
        write_manifest(output, assets, metadata, contact_timestamps)
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
