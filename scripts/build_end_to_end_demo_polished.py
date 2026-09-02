"""Build the separate polished 29-second TessScope end-to-end demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts import build_end_to_end_demo as base

PROJECT_ROOT = base.PROJECT_ROOT
OUTPUT_ROOT = base.OUTPUT_ROOT
DEFAULT_OUTPUT = OUTPUT_ROOT / "tessscope-end-to-end-demo-polished.mp4"
POSTER_PATH = OUTPUT_ROOT / "tessscope-end-to-end-demo-polished-poster.png"
CONTACT_SHEET_PATH = OUTPUT_ROOT / "tessscope-end-to-end-demo-polished-contact-sheet.png"
READABILITY_SHEET_PATH = (
    OUTPUT_ROOT / "tessscope-end-to-end-demo-polished-readability-contact-sheet.png"
)
MANIFEST_PATH = OUTPUT_ROOT / "tessscope-end-to-end-demo-polished-provenance.json"
STORYBOARD_PATH = OUTPUT_ROOT / "end-to-end-demo-polished-storyboard.md"
COPY_PATH = OUTPUT_ROOT / "end-to-end-demo-polished-on-screen-copy.md"

WIDTH = 1920
HEIGHT = 1080
FPS = 30
DURATION_SECONDS = 29
FRAME_COUNT = FPS * DURATION_SECONDS
TRANSITION_SECONDS = 0.20
SAFE_MARGIN = 72
SCENES = (
    ("hook", 0.0, 3.0),
    ("optics_hero", 3.0, 7.0),
    ("forward", 7.0, 12.0),
    ("reverse", 12.0, 16.0),
    ("correction", 16.0, 21.0),
    ("correction_result", 21.0, 25.0),
    ("causal", 25.0, 27.0),
    ("lockup", 27.0, 29.0),
)
SAMPLE_TIMESTAMPS = (1.0, 3.6, 5.9, 10.8, 15.6, 18.7, 23.0, 26.5, 28.2)

PAPER = base.PAPER
WHITE = base.WHITE
NAVY = base.NAVY
NAVY_2 = base.NAVY_2
MUTED = base.MUTED
LIGHT = base.LIGHT
CYAN = base.CYAN
BLUE = base.BLUE
MAGENTA = base.MAGENTA
ORANGE = base.ORANGE
GREEN = base.GREEN
PALE_CYAN = (232, 248, 252)
PALE_PURPLE = (246, 238, 250)
PURPLE = (125, 82, 168)

PRESERVED_VIDEO_PATHS = (
    *base.PRESERVED_VIDEO_PATHS,
    base.DEFAULT_OUTPUT,
)
EXPECTED_PRESERVED_HASHES = {
    "outputs/video/tessscope-demo.mp4": (
        "767f5d03ae859d68b4d06b94a1398beda1a38026e1549bdb7b04606254a275a6"
    ),
    "outputs/video/linkedin/tessscope-linkedin-teaser.mp4": (
        "fb0d1bfa9459d12b80b26f49f52cf960e23029758c77df6fda650a4f149e296a"
    ),
    "outputs/video/linkedin/tessscope-linkedin-fastcut.mp4": (
        "f511ec5e4609d3bd47b349aa2b2cf08b814df422511047ee6d4994e5d8635ee8"
    ),
    "outputs/video/linkedin/tessscope-end-to-end-demo.mp4": (
        "fc37230c163bd2d216d33962d3f0b8f2b13518d0bc7ddbf22c761b7f5f013658"
    ),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserved_video_hashes() -> dict[str, dict]:
    records = {}
    for path in PRESERVED_VIDEO_PATHS:
        relative = str(path.relative_to(PROJECT_ROOT))
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_path(path)
        if digest != EXPECTED_PRESERVED_HASHES[relative]:
            raise ValueError(f"Preserved video hash changed: {relative}")
        records[relative] = {
            "sha256": digest,
            "bytes": path.stat().st_size,
            "preserved": True,
        }
    return records


def _scene_base(time_seconds: float, assets: base.Assets) -> Image.Image:
    image = base._base_frame(time_seconds, assets)
    draw = ImageDraw.Draw(image)
    completed = int((WIDTH - 2 * SAFE_MARGIN) * time_seconds / DURATION_SECONDS)
    if completed > 8:
        draw.rounded_rectangle(
            (SAFE_MARGIN, 1039, SAFE_MARGIN + completed, 1057),
            radius=9,
            fill=CYAN,
        )
    return image


def _header(
    image: Image.Image,
    assets: base.Assets,
    *,
    kicker: str,
    title: str,
    accent: tuple[int, int, int],
    title_size: int = 58,
    subtitle: str | None = None,
) -> None:
    base._text(
        image,
        (960, 108),
        kicker,
        assets,
        size=28,
        color=accent,
        bold=True,
        anchor="ma",
        align="center",
        bounds=(72, 90, 1848, 142),
    )
    base._text(
        image,
        (960, 148),
        title,
        assets,
        size=title_size,
        color=NAVY,
        bold=True,
        anchor="ma",
        align="center",
        spacing=8,
        bounds=(72, 130, 1848, 272),
    )
    if subtitle:
        base._text(
            image,
            (960, 224),
            subtitle,
            assets,
            size=32,
            color=MUTED,
            anchor="ma",
            align="center",
            bounds=(110, 204, 1810, 270),
        )


def _footer(
    image: Image.Image,
    assets: base.Assets,
    primary: str,
    secondary: str | None = None,
    *,
    accent: tuple[int, int, int] = CYAN,
    primary_size: int = 40,
) -> None:
    base._panel(image, (72, 875, 1848, 1018), fill=WHITE, outline=LIGHT)
    primary_y = 921 if secondary else 945
    base._text(
        image,
        (960, primary_y),
        primary,
        assets,
        size=primary_size,
        color=accent,
        bold=True,
        anchor="ma",
        align="center",
        bounds=(105, 892, 1815, 1000),
    )
    if secondary:
        base._text(
            image,
            (960, 982),
            secondary,
            assets,
            size=30,
            color=MUTED,
            anchor="ma",
            align="center",
            bounds=(105, 960, 1815, 1016),
        )


def _accent_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    active: bool = True,
) -> None:
    fill = WHITE if not active else tuple(
        int(WHITE[index] * 0.82 + color[index] * 0.06 + 20) for index in range(3)
    )
    fill = tuple(min(255, channel) for channel in fill)
    base._panel(
        image,
        box,
        fill=fill,
        outline=color if active else LIGHT,
        width=4 if active else 3,
        radius=26,
    )


def scene_hook(local: float, assets: base.Assets, global_time: float) -> Image.Image:
    image = _scene_base(global_time, assets)
    _header(
        image,
        assets,
        kicker="TessScope · DIFFERENTIABLE CLOSED-LOOP MICROSCOPY",
        title="Can one exact gradient teach a microscope\nto refocus?",
        accent=BLUE,
        title_size=57,
    )
    _accent_panel(image, (90, 285, 860, 845), BLUE)
    base._paste_square(image, assets.object_image, (218, 310, 733, 825))

    _accent_panel(image, (960, 285, 1830, 845), PURPLE)
    base._text(
        image,
        (1395, 355),
        "BBBC006",
        assets,
        size=76,
        color=BLUE,
        bold=True,
        anchor="ma",
        bounds=(1020, 300, 1770, 445),
    )
    base._text(
        image,
        (1395, 475),
        "U2OS Hoechst nuclei",
        assets,
        size=47,
        color=NAVY,
        bold=True,
        anchor="ma",
        bounds=(1020, 430, 1770, 535),
    )
    steps = ("BIOLOGICAL INPUT", "OPTICAL DESIGN", "AUTOFOCUS ACTION")
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(steps):
        y = 590 + index * 78
        active = local >= 0.35 + index * 0.55
        draw.ellipse((1070, y - 12, 1094, y + 12), fill=CYAN if active else LIGHT)
        base._text(
            image,
            (1130, y),
            label,
            assets,
            size=32,
            color=NAVY if active else MUTED,
            bold=True,
            anchor="lm",
            bounds=(1120, y - 35, 1740, y + 35),
        )
    _footer(
        image,
        assets,
        "Real fluorescence input begins the differentiable loop",
        accent=BLUE,
    )
    return image


def _depth_text(depth: int) -> str:
    return "0 µm" if depth == 0 else f"{depth:+d} µm"


def scene_optics_hero(
    local: float, assets: base.Assets, global_time: float
) -> Image.Image:
    image = _scene_base(global_time, assets)
    _header(
        image,
        assets,
        kicker="LEARNED OPTICS",
        title="A phase pupil shapes focus across depth",
        accent=PURPLE,
    )
    _accent_panel(image, (72, 275, 800, 850), PURPLE)
    base._text(
        image,
        (436, 315),
        "EXACT B7 PHASE PUPIL",
        assets,
        size=34,
        color=PURPLE,
        bold=True,
        anchor="ma",
        bounds=(105, 290, 767, 355),
    )
    base._paste_square(image, assets.pupil, (206, 360, 666, 820))

    _accent_panel(image, (870, 275, 1848, 850), MAGENTA)
    depth_position = min(6, int(local / 4.0 * 7))
    base._text(
        image,
        (1359, 315),
        "CHROMATIX PSF SWEEP",
        assets,
        size=34,
        color=MAGENTA,
        bold=True,
        anchor="ma",
        bounds=(910, 290, 1808, 355),
    )
    base._paste_square(image, assets.psfs[depth_position], (1124, 350, 1524, 750))
    base._text(
        image,
        (1705, 495),
        _depth_text(base.DEPTHS_UM[depth_position]),
        assets,
        size=49,
        color=ORANGE,
        bold=True,
        anchor="mm",
        bounds=(1595, 430, 1825, 560),
    )
    base._text(
        image,
        (1705, 565),
        "sensor depth",
        assets,
        size=28,
        color=MUTED,
        anchor="ma",
        bounds=(1595, 535, 1825, 605),
    )

    token_width = 104
    token_gap = 16
    total = 7 * token_width + 6 * token_gap
    start_x = 870 + (978 - total) // 2
    for index, depth in enumerate(base.DEPTHS_UM):
        x0 = start_x + index * (token_width + token_gap)
        active = index == depth_position
        base._panel(
            image,
            (x0, 746, x0 + token_width, 823),
            fill=PALE_PURPLE if active else WHITE,
            outline=MAGENTA if active else LIGHT,
            width=4 if active else 2,
            radius=15,
        )
        base._text(
            image,
            (x0 + token_width // 2, 784),
            f"{depth:+d}",
            assets,
            size=27,
            color=PURPLE if active else MUTED,
            bold=active,
            anchor="mm",
        )
    _footer(
        image,
        assets,
        "Seven depths · one learned pupil",
        "Exact recorded PSFs from −6 to +6 µm",
        accent=PURPLE,
    )
    return image


SERVICE_CARDS = (
    (
        (72, 295, 620, 705),
        "OPTICS TESSERACT",
        "JAX · Chromatix",
        "T1 first exposure\nT2 corrected exposure",
        CYAN,
    ),
    (
        (686, 295, 1234, 705),
        "AUTOFOCUS TESSERACT",
        "NumPy · SciPy",
        "focus estimate\nstage action",
        GREEN,
    ),
    (
        (1300, 295, 1848, 705),
        "OBSERVER TESSERACT",
        "PyTorch · InstanSeg",
        "nuclei observation\nobjective J",
        ORANGE,
    ),
)


def scene_forward(local: float, assets: base.Assets, global_time: float) -> Image.Image:
    image = _scene_base(global_time, assets)
    _header(
        image,
        assets,
        kicker="FORWARD JOURNEY",
        title="Three Tesseracts compose one workflow",
        accent=CYAN,
    )
    active_card = min(2, int(local / 5.0 * 3))
    for index, (box, title, runtime, role, color) in enumerate(SERVICE_CARDS):
        _accent_panel(image, box, color, active=index <= active_card)
        center = (box[0] + box[2]) // 2
        base._text(
            image,
            (center, 355),
            title,
            assets,
            size=36,
            color=color,
            bold=True,
            anchor="ma",
            align="center",
            bounds=(box[0] + 20, 320, box[2] - 20, 410),
        )
        base._text(
            image,
            (center, 470),
            runtime,
            assets,
            size=43,
            color=NAVY,
            bold=True,
            anchor="ma",
            bounds=(box[0] + 20, 430, box[2] - 20, 520),
        )
        base._text(
            image,
            (center, 580),
            role,
            assets,
            size=32,
            color=MUTED,
            anchor="ma",
            align="center",
            spacing=10,
            bounds=(box[0] + 25, 535, box[2] - 25, 670),
        )
    base._arrow(
        image,
        (632, 500),
        (674, 500),
        color=CYAN,
        width=9,
        fraction=base.progress(local, 0.9, 1.8),
    )
    base._arrow(
        image,
        (1246, 500),
        (1288, 500),
        color=CYAN,
        width=9,
        fraction=base.progress(local, 2.5, 3.4),
    )
    base._panel(image, (120, 745, 1800, 835), fill=WHITE, outline=(189, 217, 230))
    base._text(
        image,
        (960, 790),
        "INPUT → T1 → AUTOFOCUS → STAGE → T2 → OBSERVER → J",
        assets,
        size=37,
        color=NAVY_2,
        bold=True,
        anchor="mm",
        bounds=(155, 755, 1765, 825),
    )
    _footer(
        image,
        assets,
        "Three Tesseracts · three native runtimes · one differentiable workflow",
        "The optics service forms both the first and corrected exposures",
        accent=CYAN,
        primary_size=37,
    )
    return image


REVERSE_NODES = (
    ((72, 335, 355, 625), "PUPIL UPDATE", "θ5 … θ11", PURPLE),
    ((395, 335, 690, 625), "OPTICS T1", "first exposure", CYAN),
    ((730, 335, 1050, 625), "AUTOFOCUS", "action", GREEN),
    ((1090, 335, 1385, 625), "OPTICS T2", "corrected", CYAN),
    ((1425, 335, 1720, 625), "OBSERVER", "PyTorch", ORANGE),
)


def scene_reverse(local: float, assets: base.Assets, global_time: float) -> Image.Image:
    image = _scene_base(global_time, assets)
    _header(
        image,
        assets,
        kicker="EXACT REVERSE GRADIENT",
        title="One gradient crosses every boundary",
        accent=PURPLE,
    )
    overall = min(1.0, local / 3.35)
    active_count = int(overall * 6)
    for index, (box, title, role, color) in enumerate(REVERSE_NODES):
        reverse_rank = len(REVERSE_NODES) - index
        active = active_count >= reverse_rank
        _accent_panel(image, box, color, active=active)
        center = (box[0] + box[2]) // 2
        base._text(
            image,
            (center, 420),
            title,
            assets,
            size=31,
            color=color,
            bold=True,
            anchor="ma",
            align="center",
            bounds=(box[0] + 15, 375, box[2] - 15, 470),
        )
        base._text(
            image,
            (center, 525),
            role,
            assets,
            size=31,
            color=NAVY_2,
            anchor="ma",
            bounds=(box[0] + 15, 490, box[2] - 15, 570),
        )

    draw = ImageDraw.Draw(image)
    draw.ellipse((1752, 418, 1848, 514), fill=PALE_CYAN, outline=BLUE, width=4)
    base._text(
        image,
        (1800, 466),
        "J",
        assets,
        size=46,
        color=BLUE,
        bold=True,
        anchor="mm",
    )
    path = [
        ((1750, 466), (1728, 466)),
        ((1417, 466), (1393, 466)),
        ((1082, 466), (1058, 466)),
        ((722, 466), (698, 466)),
        ((387, 466), (363, 466)),
    ]
    for index, (start, end) in enumerate(path):
        base._arrow(
            image,
            start,
            end,
            color=PURPLE,
            width=9,
            fraction=base.progress(overall, index / 5.0, (index + 1) / 5.0),
        )

    token_width = 124
    gap = 22
    total = 7 * token_width + 6 * gap
    start_x = (WIDTH - total) // 2
    for index in range(7):
        revealed = base.progress(local, 2.55 + index * 0.07, 2.9 + index * 0.07)
        x0 = start_x + index * (token_width + gap)
        base._panel(
            image,
            (x0, 690, x0 + token_width, 790),
            fill=PALE_PURPLE if revealed > 0.5 else WHITE,
            outline=PURPLE if revealed > 0.5 else LIGHT,
            width=4 if revealed > 0.5 else 2,
            radius=18,
        )
        base._text(
            image,
            (x0 + token_width // 2, 740),
            f"θ{index + 5}",
            assets,
            size=36,
            color=PURPLE if revealed > 0.5 else MUTED,
            bold=True,
            anchor="mm",
        )
    _footer(
        image,
        assets,
        "OBJECTIVE J → OBSERVER → OPTICS T2 → AUTOFOCUS ACTION → OPTICS T1",
        "Seven phase parameters update end-to-end",
        accent=PURPLE,
        primary_size=34,
    )
    return image


def scene_correction(local: float, assets: base.Assets, global_time: float) -> Image.Image:
    image = _scene_base(global_time, assets)
    _header(
        image,
        assets,
        kicker="AUTOFOCUS CORRECTION",
        title="See defocus → predict stage action → expose again",
        accent=BLUE,
        title_size=52,
    )
    _accent_panel(image, (72, 285, 650, 845), ORANGE)
    base._text(
        image,
        (361, 325),
        "FIRST EXPOSURE",
        assets,
        size=36,
        color=ORANGE,
        bold=True,
        anchor="ma",
        bounds=(100, 295, 620, 365),
    )
    base._paste_square(image, assets.before, (126, 375, 596, 845))

    _accent_panel(image, (720, 285, 1200, 845), MAGENTA)
    base._text(
        image,
        (960, 355),
        "FOCUS ESTIMATE",
        assets,
        size=32,
        color=MAGENTA,
        bold=True,
        anchor="ma",
    )
    base._text(
        image,
        (960, 500),
        f"{assets.predicted_depth_um:+.2f} µm",
        assets,
        size=76,
        color=PURPLE,
        bold=True,
        anchor="mm",
    )
    base._text(
        image,
        (960, 565),
        "focus offset",
        assets,
        size=30,
        color=MUTED,
        anchor="ma",
    )
    base._arrow(
        image,
        (815, 670),
        (1105, 670),
        color=MAGENTA,
        width=12,
        fraction=base.progress(local, 0.8, 2.1),
    )
    base._text(
        image,
        (960, 755),
        "STAGE ACTION",
        assets,
        size=34,
        color=MAGENTA,
        bold=True,
        anchor="ma",
    )

    corrected_active = local >= 2.0
    _accent_panel(image, (1270, 285, 1848, 845), BLUE, active=corrected_active)
    base._text(
        image,
        (1559, 325),
        "CORRECTED EXPOSURE",
        assets,
        size=36,
        color=BLUE,
        bold=True,
        anchor="ma",
        bounds=(1290, 295, 1828, 365),
    )
    base._paste_square(image, assets.corrected, (1324, 375, 1794, 845))
    base._arrow(
        image,
        (662, 565),
        (708, 565),
        color=ORANGE,
        width=9,
        fraction=base.progress(local, 0.25, 0.9),
    )
    base._arrow(
        image,
        (1212, 565),
        (1258, 565),
        color=BLUE,
        width=9,
        fraction=base.progress(local, 1.75, 2.5),
    )
    _footer(
        image,
        assets,
        "Matched −6 µm field · identical display transform",
        accent=BLUE,
    )
    return image


def scene_correction_result(
    local: float, assets: base.Assets, global_time: float
) -> Image.Image:
    image = _scene_base(global_time, assets)
    _header(
        image,
        assets,
        kicker="CORRECTION OUTCOME",
        title="Hard off-focus frames improve after stage correction",
        accent=BLUE,
        title_size=52,
    )
    _accent_panel(image, (72, 285, 930, 845), BLUE)
    base._text(
        image,
        (501, 325),
        "MATCHED BBBC006 FIELD",
        assets,
        size=32,
        color=MUTED,
        bold=True,
        anchor="ma",
    )
    base._paste_square(image, assets.before, (115, 390, 485, 760))
    base._paste_square(image, assets.corrected, (517, 390, 887, 760))
    base._arrow(image, (489, 575), (513, 575), color=BLUE, width=8)
    base._text(image, (300, 800), "FIRST", assets, size=30, color=ORANGE, bold=True, anchor="ma")
    base._text(image, (702, 800), "CORRECTED", assets, size=30, color=BLUE, bold=True, anchor="ma")

    _accent_panel(image, (990, 285, 1848, 845), BLUE)
    animated_after = base.EXPECTED_CLAIMS["before_pq"] + (
        base.EXPECTED_CLAIMS["after_pq"] - base.EXPECTED_CLAIMS["before_pq"]
    ) * base.progress(local, 0.25, 1.2)
    base._text(
        image,
        (1419, 395),
        f"PQ 0.4922 → {animated_after:.4f}",
        assets,
        size=64,
        color=BLUE,
        bold=True,
        anchor="ma",
        bounds=(1020, 345, 1818, 470),
    )
    base._text(
        image,
        (1419, 565),
        "74.7%",
        assets,
        size=112,
        color=GREEN,
        bold=True,
        anchor="mm",
        bounds=(1060, 480, 1780, 660),
    )
    base._text(
        image,
        (1419, 690),
        "of hard frames improve",
        assets,
        size=42,
        color=NAVY,
        bold=True,
        anchor="ma",
        bounds=(1050, 650, 1790, 745),
    )
    base._text(
        image,
        (1419, 785),
        "27 hard-density wells",
        assets,
        size=34,
        color=MUTED,
        anchor="ma",
    )
    _footer(
        image,
        assets,
        "PQ 0.4922 → 0.5523",
        "74.7% of hard frames improve",
        accent=BLUE,
    )
    return image


def _value_to_x(value: float) -> int:
    minimum = -0.002
    maximum = 0.013
    return int(round(860 + (value - minimum) / (maximum - minimum) * (1740 - 860)))


def scene_causal(local: float, assets: base.Assets, global_time: float) -> Image.Image:
    image = _scene_base(global_time, assets)
    _header(
        image,
        assets,
        kicker="CAUSAL GRADIENT EVIDENCE",
        title="Exact vs forward-identical stopped gradient",
        accent=PURPLE,
        title_size=53,
    )
    _accent_panel(image, (72, 285, 750, 835), PURPLE)
    base._text(
        image,
        (411, 420),
        "+0.0063 PQ",
        assets,
        size=82,
        color=PURPLE,
        bold=True,
        anchor="mm",
        bounds=(110, 335, 712, 515),
    )
    base._text(
        image,
        (411, 570),
        "95% CI",
        assets,
        size=37,
        color=MUTED,
        bold=True,
        anchor="ma",
    )
    base._text(
        image,
        (411, 640),
        "[+0.0007, +0.0115]",
        assets,
        size=40,
        color=MAGENTA,
        bold=True,
        anchor="ma",
    )
    base._text(
        image,
        (411, 755),
        "paired difference in PQ",
        assets,
        size=30,
        color=MUTED,
        anchor="ma",
    )

    _accent_panel(image, (800, 285, 1848, 835), PURPLE)
    draw = ImageDraw.Draw(image)
    axis_y = 585
    draw.line((860, axis_y, 1740, axis_y), fill=NAVY_2, width=5)
    for tick in (-0.002, 0.0, 0.004, 0.008, 0.012):
        x = _value_to_x(tick)
        draw.line((x, axis_y - 14, x, axis_y + 14), fill=NAVY_2, width=4)
        label = "0" if tick == 0 else f"{tick:+.3f}"
        base._text(
            image,
            (x, 628),
            label,
            assets,
            size=27,
            color=MUTED,
            anchor="ma",
        )
    zero_x = _value_to_x(0.0)
    draw.line((zero_x, 430, zero_x, 700), fill=LIGHT, width=4)
    low = _value_to_x(base.EXPECTED_CLAIMS["causal_ci_lower_95"])
    high = _value_to_x(base.EXPECTED_CLAIMS["causal_ci_upper_95"])
    point = _value_to_x(base.EXPECTED_CLAIMS["causal_gain"])
    reveal = base.progress(local, 0.12, 0.95)
    animated_high = int(low + (high - low) * reveal)
    draw.line((low, axis_y, animated_high, axis_y), fill=MAGENTA, width=20)
    if reveal > 0.95:
        draw.line((low, axis_y - 36, low, axis_y + 36), fill=MAGENTA, width=7)
        draw.line((high, axis_y - 36, high, axis_y + 36), fill=MAGENTA, width=7)
        draw.ellipse((point - 20, axis_y - 20, point + 20, axis_y + 20), fill=BLUE)
    base._text(
        image,
        (1324, 390),
        "EXACT − STOPPED",
        assets,
        size=36,
        color=PURPLE,
        bold=True,
        anchor="ma",
    )
    base._text(
        image,
        (1324, 745),
        "forward-identical control",
        assets,
        size=34,
        color=NAVY,
        bold=True,
        anchor="ma",
    )
    _footer(
        image,
        assets,
        "27 hard-density wells · 2,000 grouped bootstrap replicates",
        accent=PURPLE,
        primary_size=36,
    )
    return image


def scene_lockup(local: float, assets: base.Assets, global_time: float) -> Image.Image:
    image = _scene_base(global_time, assets)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 450, 82), fill=WHITE)
    base._text(
        image,
        (960, 120),
        "TessScope",
        assets,
        size=104,
        color=NAVY,
        bold=True,
        anchor="ma",
        bounds=(520, 85, 1400, 240),
    )
    base._text(
        image,
        (960, 245),
        "Differentiable closed-loop microscopy",
        assets,
        size=46,
        color=BLUE,
        bold=True,
        anchor="ma",
        bounds=(380, 205, 1540, 305),
    )

    _accent_panel(image, (220, 330, 650, 760), PURPLE)
    base._paste_square(image, assets.pupil, (280, 390, 590, 700))
    _accent_panel(image, (1270, 330, 1700, 760), MAGENTA)
    psf_index = min(6, int(local / 2.0 * 7))
    base._paste_square(image, assets.psfs[psf_index], (1330, 390, 1640, 700))

    center_x = 960
    draw.line((680, 545, 1240, 545), fill=CYAN, width=9)
    moving_x = 680 + int(560 * ((local / 1.15) % 1.0))
    draw.ellipse((moving_x - 14, 531, moving_x + 14, 559), fill=PURPLE)
    base._text(
        image,
        (center_x, 460),
        "EXACT",
        assets,
        size=42,
        color=PURPLE,
        bold=True,
        anchor="mm",
    )
    base._text(
        image,
        (center_x, 625),
        "GRADIENT",
        assets,
        size=42,
        color=PURPLE,
        bold=True,
        anchor="mm",
    )
    base._text(
        image,
        (960, 815),
        "Tesseract Hackathon 2026 · Track 05",
        assets,
        size=48,
        color=NAVY,
        bold=True,
        anchor="ma",
        bounds=(350, 770, 1570, 865),
    )
    _footer(
        image,
        assets,
        "Exact gradients across JAX · NumPy/SciPy · PyTorch",
        accent=CYAN,
        primary_size=38,
    )
    return image


SCENE_RENDERERS = (
    scene_hook,
    scene_optics_hero,
    scene_forward,
    scene_reverse,
    scene_correction,
    scene_correction_result,
    scene_causal,
    scene_lockup,
)


def render_frame(time_seconds: float, assets: base.Assets) -> Image.Image:
    if not 0.0 <= time_seconds < DURATION_SECONDS:
        raise ValueError("Frame time must fall inside the 29-second duration")
    index = next(
        index
        for index, (_, start, end) in enumerate(SCENES)
        if start <= time_seconds < end
    )
    _, start, end = SCENES[index]
    current = SCENE_RENDERERS[index](time_seconds - start, assets, time_seconds)
    if index < len(SCENES) - 1 and time_seconds >= end - TRANSITION_SECONDS:
        transition = base.progress(time_seconds, end - TRANSITION_SECONDS, end)
        following = SCENE_RENDERERS[index + 1](
            transition * TRANSITION_SECONDS,
            assets,
            time_seconds,
        )
        split = int(WIDTH * transition)
        if split > 0:
            current.alpha_composite(following.crop((0, 0, split, HEIGHT)), (0, 0))
            ImageDraw.Draw(current).line((split, 0, split, HEIGHT), fill=PURPLE, width=10)
    return current.convert("RGB")


def project_path(value: Path) -> Path:
    path = value if value.is_absolute() else PROJECT_ROOT / value
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Polished demo output must stay inside the project")
    return path


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
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-fps_mode",
        "cfr",
        "-frames:v",
        str(FRAME_COUNT),
        "-threads",
        "1",
        "-x264-params",
        "keyint=60:min-keyint=60:scenecut=0",
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-metadata",
        "title=TessScope — polished 29-second end-to-end demo",
        "-metadata",
        "comment=Frozen validation scientific assets; intentional no-audio stream",
        str(output),
    ]


def encode_video(ffmpeg: str, output: Path, assets: base.Assets) -> None:
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
        raise RuntimeError("FFmpeg stopped while receiving frames") from error
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)


def _stamp(image: Image.Image, timestamp: float, assets: base.Assets, *, small: bool) -> None:
    draw = ImageDraw.Draw(image)
    if small:
        box = (505, 320, 632, 354)
        selected_font = base._font(assets, 18, bold=True)
        xy = (620, 337)
    else:
        box = (1515, 960, 1895, 1052)
        selected_font = base._font(assets, 42, bold=True)
        xy = (1865, 1006)
    draw.rounded_rectangle(box, radius=12 if small else 24, fill=NAVY)
    draw.text(xy, f"{timestamp:05.2f}s", font=selected_font, fill=WHITE, anchor="rm")


def create_review_assets(
    ffmpeg: str, output: Path, assets: base.Assets
) -> list[dict]:
    review_root = PROJECT_ROOT / ".release-check" / "end-to-end-demo-polished-frames"
    review_root.mkdir(parents=True, exist_ok=True)
    frames = []
    records = []
    for index, timestamp in enumerate(SAMPLE_TIMESTAMPS):
        frame_index = min(FRAME_COUNT - 1, round(timestamp * FPS))
        path = review_root / f"frame-{index + 1:02d}-{frame_index:04d}.png"
        base.extract_frame(ffmpeg, output, frame_index, path)
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
        records.append(
            {
                "timestamp_seconds": timestamp,
                "frame_index": frame_index,
                "encoded_frame_sha256": sha256_path(path),
            }
        )

    full_sheet = Image.new("RGB", (WIDTH * 3, HEIGHT * 3), PAPER)
    small_sheet = Image.new("RGB", (640 * 3, 360 * 3), PAPER)
    for index, (timestamp, frame) in enumerate(zip(SAMPLE_TIMESTAMPS, frames, strict=True)):
        row, column = divmod(index, 3)
        full = frame.copy()
        _stamp(full, timestamp, assets, small=False)
        full_sheet.paste(full, (column * WIDTH, row * HEIGHT))
        small = frame.resize((640, 360), Image.Resampling.LANCZOS)
        _stamp(small, timestamp, assets, small=True)
        small_sheet.paste(small, (column * 640, row * 360))
    full_sheet.save(CONTACT_SHEET_PATH, optimize=False)
    small_sheet.save(READABILITY_SHEET_PATH, optimize=False)
    base.extract_frame(ffmpeg, output, round(5.9 * FPS), POSTER_PATH)
    return records


def write_manifest(
    output: Path,
    assets: base.Assets,
    frame_records: list[dict],
    preserved: dict[str, dict],
    metadata: dict,
) -> None:
    claims = base.verify_claim_sources(assets)
    selected_patch = next(
        patch for patch in assets.metadata["patches"] if patch["field_id"] == "n21_s1"
    )
    sources = (
        base.REPLAY_PATH,
        base.REPLAY_METADATA_PATH,
        base.VALIDATION_PATH,
        base.DERIVATIVE_PATH,
        base.VISUAL_CONTRACT_PATH,
        PROJECT_ROOT / selected_patch["object_source"],
    )
    artifacts = {
        "poster": POSTER_PATH,
        "full_resolution_contact_sheet": CONTACT_SHEET_PATH,
        "readability_contact_sheet": READABILITY_SHEET_PATH,
        "storyboard": STORYBOARD_PATH,
        "on_screen_copy": COPY_PATH,
    }
    manifest = {
        "schema_version": 1,
        "status": "complete_polished_review_candidate",
        "selected_or_published": False,
        "external_actions_performed": [],
        "global_on_screen_qualifier": "FROZEN VALIDATION",
        "format": {
            "dimensions": [WIDTH, HEIGHT],
            "fps": FPS,
            "duration_seconds": DURATION_SECONDS,
            "frame_count": FRAME_COUNT,
            "codec": "H.264",
            "pixel_format": "yuv420p",
            "audio_streams": 0,
            "fast_start": True,
            "ffprobe": metadata,
        },
        "timeline": [
            {"id": name, "start_seconds": start, "end_seconds": end}
            for name, start, end in SCENES
        ],
        "scientific_media_policy": {
            "source": "frozen traced TessScope artifacts only",
            "generated_or_ai_scientific_imagery": False,
            "browser_or_website_screenshots": False,
            "per_image_normalization": False,
            "sharpening": False,
            "denoising": False,
            "scientific_interpolation": False,
        },
        "claims": {
            "correction_outcome": {
                "display": [
                    "PQ 0.4922 → 0.5523",
                    "74.7% of hard frames improve",
                ],
                "raw": {
                    "before_pq": claims["before_pq"],
                    "after_pq": claims["after_pq"],
                    "improved_fraction": claims["improved_fraction"],
                },
                "source": str(base.VALIDATION_PATH.relative_to(PROJECT_ROOT)),
            },
            "causal_gradient_evidence": {
                "display": [
                    "+0.0063 PQ",
                    "95% CI [+0.0007, +0.0115]",
                    "Exact vs forward-identical stopped gradient",
                ],
                "raw": {
                    "mean_difference": claims["causal_gain"],
                    "ci_lower_95": claims["causal_ci_lower_95"],
                    "ci_upper_95": claims["causal_ci_upper_95"],
                    "source_count": claims["source_count"],
                    "replicates": claims["bootstrap_replicates"],
                },
                "source": str(base.VALIDATION_PATH.relative_to(PROJECT_ROOT)),
            },
        },
        "scientific_sources": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(path),
            }
            for path in sources
        ],
        "scientific_transforms": {
            "input": "clip-only display of frozen replay object_image",
            "microscopy": assets.metadata["observer"]["transform"],
            "boundaries": "unchanged overlay_boundaries geometry and scale",
            "pupil": "exact-design wrapped phase; fixed twilight colormap",
            "psfs": "seven exact arrays; fixed global maximum/log floor/magma colormap",
        },
        "preserved_videos": preserved,
        "outputs": {
            "video": {
                "path": str(output.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(output),
                "bytes": output.stat().st_size,
            },
            **{
                name: {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_path(path),
                }
                for name, path in artifacts.items()
            },
        },
        "review_samples": frame_records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise SystemExit("FFmpeg and ffprobe are required")
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    preserved_before = preserved_video_hashes()
    assets = base.load_assets()
    encode_video(ffmpeg, output, assets)
    preserved_after = preserved_video_hashes()
    if preserved_before != preserved_after:
        raise ValueError("A preserved video changed during the polished build")
    metadata = base.probe(ffprobe, output)
    if output == DEFAULT_OUTPUT.resolve():
        records = create_review_assets(ffmpeg, output, assets)
        write_manifest(output, assets, records, preserved_after, metadata)
        result = {
            "status": "rendered_polished_review_candidate",
            "output": str(output.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(output),
            "frames": FRAME_COUNT,
            "review_samples": len(records),
        }
    else:
        result = {
            "status": "rendered_noncanonical_comparison",
            "output": str(output.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(output),
            "ffprobe": metadata,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
