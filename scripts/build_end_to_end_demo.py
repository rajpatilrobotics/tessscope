"""Build the purpose-made 29-second TessScope end-to-end demo video."""

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
from PIL import Image, ImageDraw, ImageFont

from tessscope.demo.figures import overlay_boundaries

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "video" / "linkedin"
DEFAULT_OUTPUT = OUTPUT_ROOT / "tessscope-end-to-end-demo.mp4"
POSTER_PATH = OUTPUT_ROOT / "tessscope-end-to-end-demo-poster.png"
CONTACT_SHEET_PATH = OUTPUT_ROOT / "tessscope-end-to-end-demo-contact-sheet.png"
READABILITY_SHEET_PATH = (
    OUTPUT_ROOT / "tessscope-end-to-end-demo-readability-contact-sheet.png"
)
MANIFEST_PATH = OUTPUT_ROOT / "tessscope-end-to-end-demo-provenance.json"
STORYBOARD_PATH = OUTPUT_ROOT / "end-to-end-demo-storyboard.md"
COPY_PATH = OUTPUT_ROOT / "end-to-end-demo-on-screen-copy.md"

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

PRESERVED_VIDEO_PATHS = (
    PROJECT_ROOT / "outputs" / "video" / "tessscope-demo.mp4",
    OUTPUT_ROOT / "tessscope-linkedin-teaser.mp4",
    OUTPUT_ROOT / "tessscope-linkedin-fastcut.mp4",
)

WIDTH = 1920
HEIGHT = 1080
FPS = 30
DURATION_SECONDS = 29
FRAME_COUNT = FPS * DURATION_SECONDS
SAFE_MARGIN = 72
TRANSITION_SECONDS = 0.22
DEPTHS_UM = (-6, -4, -2, 0, 2, 4, 6)
SCENES = (
    ("input", 0.0, 3.0),
    ("optics", 3.0, 7.0),
    ("forward", 7.0, 11.5),
    ("reverse", 11.5, 15.5),
    ("correction", 15.5, 20.5),
    ("stage_result", 20.5, 24.5),
    ("causal", 24.5, 27.0),
    ("lockup", 27.0, 29.0),
)
SAMPLE_TIMESTAMPS = (1.0, 4.8, 8.4, 14.8, 17.0, 19.5, 22.5, 25.5, 28.2)

PAPER = (247, 249, 252)
WHITE = (255, 255, 255)
NAVY = (8, 24, 45)
NAVY_2 = (19, 43, 69)
MUTED = (88, 107, 126)
LIGHT = (222, 230, 238)
PALE_CYAN = (227, 248, 252)
PALE_MAGENTA = (249, 235, 249)
CYAN = (0, 151, 190)
BLUE = (0, 114, 178)
MAGENTA = (204, 121, 167)
ORANGE = (213, 94, 0)
GREEN = (0, 158, 115)

DESIGN_NAME = "v2_3-exact-balanced-segmentation_only-step-14"
EXPECTED_REPLAY_SHA256 = (
    "586b02362c2aa4c0dc31633ac2de661d1d6fb20ce24524088fff81fefc477a04"
)
EXPECTED_CLAIMS = {
    "before_pq": 0.4921920085941332,
    "after_pq": 0.552294837627718,
    "improved_fraction": 0.7469135802469136,
    "causal_gain": 0.006258840610359453,
    "causal_ci_lower_95": 0.0006703181908061804,
    "causal_ci_upper_95": 0.011547458540670894,
    "source_count": 27,
    "bootstrap_replicates": 2000,
}


@dataclass(frozen=True)
class Assets:
    """Scientific images and frozen evidence used by the renderer."""

    object_image: Image.Image
    before: Image.Image
    corrected: Image.Image
    pupil: Image.Image
    psfs: tuple[Image.Image, ...]
    metadata: dict
    validation: dict
    predicted_depth_um: float
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


def _gray(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    pixels = np.round(clipped * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(pixels[..., None], 3, axis=-1), mode="RGB")


def load_assets() -> Assets:
    metadata = _json(REPLAY_METADATA_PATH)
    validation = _json(VALIDATION_PATH)
    if sha256_path(REPLAY_PATH) != EXPECTED_REPLAY_SHA256:
        raise ValueError("Frozen replay archive hash changed")
    if metadata["archive"]["sha256"] != EXPECTED_REPLAY_SHA256:
        raise ValueError("Replay metadata no longer points to the frozen archive")

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
        phase = np.angle(np.exp(1j * replay["pupil_phase_radians"][exact_index]))
        phase_rgba = np.zeros((*phase.shape, 4), dtype=np.uint8)
        colors = colormaps["twilight"]((phase + np.pi) / (2 * np.pi))
        phase_rgba[..., :3] = np.round(colors[..., :3] * 255).astype(np.uint8)
        phase_rgba[..., 3] = mask.astype(np.uint8) * 255

        psf_stack = replay["psf_sensor"][exact_index]
        global_maximum = float(replay["psf_sensor"].max())
        crop_start = (psf_stack.shape[-1] - 41) // 2
        psfs: list[Image.Image] = []
        for psf in psf_stack:
            cropped = psf[crop_start : crop_start + 41, crop_start : crop_start + 41]
            log_psf = np.log10(np.maximum(cropped / global_maximum, 1e-5))
            psf_colors = colormaps["magma"]((log_psf + 5.0) / 5.0)[..., :3]
            psfs.append(_rgb(psf_colors))

        object_image = _gray(replay["object_image"])
        predicted_depth = float(replay["predicted_depth_um"][exact_index, depth_index])

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
    assets = Assets(
        object_image=object_image,
        before=_rgb(before),
        corrected=_rgb(corrected),
        pupil=Image.fromarray(phase_rgba, mode="RGBA"),
        psfs=tuple(psfs),
        metadata=metadata,
        validation=validation,
        predicted_depth_um=predicted_depth,
        font_regular_path=regular,
        font_bold_path=bold,
    )
    verify_claim_sources(assets)
    return assets


@lru_cache(maxsize=96)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _font(assets: Assets, size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = assets.font_bold_path if bold else assets.font_regular_path
    return font(str(path), size)


def _contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _text(
    image: Image.Image,
    xy: tuple[int, int],
    value: str,
    assets: Assets,
    *,
    size: int,
    color: tuple[int, int, int] = NAVY,
    bold: bool = False,
    anchor: str = "la",
    align: str = "left",
    spacing: int = 8,
    bounds: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    draw = ImageDraw.Draw(image)
    selected_font = _font(assets, size, bold=bold)
    bbox = draw.multiline_textbbox(
        xy,
        value,
        font=selected_font,
        anchor=anchor,
        align=align,
        spacing=spacing,
    )
    if bounds is not None and not _contains(bounds, bbox):
        raise ValueError(f"Text escaped its layout region: {value!r} {bbox} not in {bounds}")
    draw.multiline_text(
        xy,
        value,
        font=selected_font,
        fill=color,
        anchor=anchor,
        align=align,
        spacing=spacing,
    )
    return bbox


def _panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int] = WHITE,
    outline: tuple[int, int, int] = LIGHT,
    width: int = 3,
    radius: int = 24,
) -> None:
    ImageDraw.Draw(image).rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def _arrow(
    image: Image.Image,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int],
    width: int = 7,
    fraction: float = 1.0,
) -> None:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0.001:
        return
    x0, y0 = start
    x1, y1 = end
    x = int(round(x0 + (x1 - x0) * fraction))
    y = int(round(y0 + (y1 - y0) * fraction))
    draw = ImageDraw.Draw(image)
    draw.line((x0, y0, x, y), fill=color, width=width)
    if fraction >= 0.97:
        angle = math.atan2(y1 - y0, x1 - x0)
        length = 20
        spread = 0.58
        points = [
            (x1, y1),
            (
                int(x1 - length * math.cos(angle - spread)),
                int(y1 - length * math.sin(angle - spread)),
            ),
            (
                int(x1 - length * math.cos(angle + spread)),
                int(y1 - length * math.sin(angle + spread)),
            ),
        ]
        draw.polygon(points, fill=color)


def _paste_square(
    image: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
    *,
    resampling: Image.Resampling = Image.Resampling.NEAREST,
) -> None:
    x0, y0, x1, y1 = box
    if x1 - x0 != y1 - y0:
        raise ValueError("Scientific panels must preserve square geometry")
    resized = source.resize((x1 - x0, y1 - y0), resampling).convert("RGBA")
    image.alpha_composite(resized, (x0, y0))
    ImageDraw.Draw(image).rectangle(box, outline=NAVY_2, width=3)


def _base_frame(time_seconds: float, assets: Assets) -> Image.Image:
    image = Image.new("RGBA", (WIDTH, HEIGHT), PAPER + (255,))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 84), fill=WHITE)
    draw.line((SAFE_MARGIN, 84, WIDTH - SAFE_MARGIN, 84), fill=LIGHT, width=2)
    _text(
        image,
        (SAFE_MARGIN, 42),
        "TessScope",
        assets,
        size=30,
        bold=True,
        anchor="lm",
        bounds=(SAFE_MARGIN, 10, 400, 75),
    )
    _text(
        image,
        (WIDTH - SAFE_MARGIN, 42),
        "FROZEN VALIDATION",
        assets,
        size=26,
        color=MUTED,
        bold=True,
        anchor="rm",
        bounds=(1450, 10, WIDTH - SAFE_MARGIN, 75),
    )
    draw.line((SAFE_MARGIN, 1048, WIDTH - SAFE_MARGIN, 1048), fill=LIGHT, width=4)
    completed = SAFE_MARGIN + int(
        (WIDTH - 2 * SAFE_MARGIN) * np.clip(time_seconds / DURATION_SECONDS, 0.0, 1.0)
    )
    draw.line((SAFE_MARGIN, 1048, completed, 1048), fill=CYAN, width=6)
    for _, start, _ in SCENES:
        x = SAFE_MARGIN + int((WIDTH - 2 * SAFE_MARGIN) * start / DURATION_SECONDS)
        draw.ellipse((x - 5, 1043, x + 5, 1053), fill=NAVY)
    return image


def _header(
    image: Image.Image,
    assets: Assets,
    *,
    kicker: str,
    title: str,
    subtitle: str | None = None,
    accent: tuple[int, int, int] = CYAN,
    title_size: int = 58,
) -> None:
    _text(
        image,
        (WIDTH // 2, 112),
        kicker,
        assets,
        size=28,
        color=accent,
        bold=True,
        anchor="ma",
        align="center",
        bounds=(SAFE_MARGIN, 94, WIDTH - SAFE_MARGIN, 142),
    )
    _text(
        image,
        (WIDTH // 2, 151),
        title,
        assets,
        size=title_size,
        bold=True,
        anchor="ma",
        align="center",
        bounds=(SAFE_MARGIN, 137, WIDTH - SAFE_MARGIN, 222),
    )
    if subtitle:
        _text(
            image,
            (WIDTH // 2, 222),
            subtitle,
            assets,
            size=32,
            color=MUTED,
            anchor="ma",
            align="center",
            bounds=(SAFE_MARGIN, 205, WIDTH - SAFE_MARGIN, 265),
        )


def _footer(
    image: Image.Image,
    assets: Assets,
    primary: str,
    secondary: str,
    *,
    accent: tuple[int, int, int] = CYAN,
) -> None:
    _panel(image, (SAFE_MARGIN, 873, WIDTH - SAFE_MARGIN, 1018), fill=WHITE)
    _text(
        image,
        (WIDTH // 2, 916),
        primary,
        assets,
        size=40,
        color=accent,
        bold=True,
        anchor="ma",
        align="center",
        bounds=(95, 892, 1825, 968),
    )
    _text(
        image,
        (WIDTH // 2, 978),
        secondary,
        assets,
        size=30,
        color=MUTED,
        anchor="ma",
        align="center",
        bounds=(95, 958, 1825, 1016),
    )


def scene_input(local: float, assets: Assets, global_time: float) -> Image.Image:
    image = _base_frame(global_time, assets)
    _header(
        image,
        assets,
        kicker="TRACED BBBC006 INPUT",
        title="A real validation field enters the loop",
        accent=BLUE,
    )
    _panel(image, (90, 265, 855, 840), fill=WHITE, outline=(182, 210, 228))
    _paste_square(image, assets.object_image, (212, 300, 732, 820))
    draw = ImageDraw.Draw(image)
    reveal = progress(local, 0.15, 1.15)
    x = 212 + int(520 * reveal)
    draw.line((x, 300, x, 820), fill=CYAN, width=8)

    _panel(image, (955, 265, 1830, 840), fill=WHITE, outline=(182, 210, 228))
    _text(
        image,
        (1392, 350),
        "Can a microscope learn\nto refocus itself?",
        assets,
        size=58,
        bold=True,
        anchor="ma",
        align="center",
        spacing=14,
        bounds=(1010, 315, 1775, 520),
    )
    _text(
        image,
        (1392, 595),
        "U2OS Hoechst",
        assets,
        size=43,
        color=BLUE,
        bold=True,
        anchor="ma",
        bounds=(1010, 560, 1775, 640),
    )
    _text(
        image,
        (1392, 672),
        "frozen validation field n21_s1",
        assets,
        size=34,
        color=MUTED,
        anchor="ma",
        bounds=(1010, 645, 1775, 715),
    )
    _text(
        image,
        (1392, 764),
        "REAL DATA  ·  TRACED SOURCE",
        assets,
        size=30,
        color=ORANGE,
        bold=True,
        anchor="ma",
        bounds=(1010, 735, 1775, 800),
    )
    _footer(
        image,
        assets,
        "Input: fluorescence nuclei from BBBC006",
        "Validation-only cached replay · not a live microscope feed",
        accent=BLUE,
    )
    return image


def scene_optics(local: float, assets: Assets, global_time: float) -> Image.Image:
    image = _base_frame(global_time, assets)
    _header(
        image,
        assets,
        kicker="DIFFERENTIABLE IMAGE FORMATION",
        title="Phase pupil → PSF → first exposure",
        accent=MAGENTA,
    )
    boxes = ((72, 280, 592, 825), (700, 280, 1220, 825), (1328, 280, 1848, 825))
    labels = ("LEARNED B7 PHASE PUPIL", "SEVEN-DEPTH PSF", "FIRST EXPOSURE")
    for box, label in zip(boxes, labels, strict=True):
        _panel(image, box, fill=WHITE)
        _text(
            image,
            ((box[0] + box[2]) // 2, 315),
            label,
            assets,
            size=31,
            color=MAGENTA if box != boxes[2] else BLUE,
            bold=True,
            anchor="ma",
            align="center",
            bounds=(box[0] + 18, 292, box[2] - 18, 345),
        )
    _paste_square(image, assets.pupil, (142, 365, 522, 745))
    psf_index = min(6, int(np.clip(local / 4.0, 0.0, 0.999) * 7))
    _paste_square(image, assets.psfs[psf_index], (770, 365, 1150, 745))
    _paste_square(image, assets.before, (1398, 365, 1778, 745))
    _text(
        image,
        (960, 775),
        f"{DEPTHS_UM[psf_index]:+d} µm",
        assets,
        size=32,
        color=ORANGE,
        bold=True,
        anchor="ma",
        bounds=(780, 750, 1140, 820),
    )
    _arrow(
        image,
        (610, 552),
        (682, 552),
        color=MAGENTA,
        fraction=progress(local, 0.45, 1.3),
    )
    _arrow(
        image,
        (1238, 552),
        (1310, 552),
        color=BLUE,
        fraction=progress(local, 1.8, 2.8),
    )
    _footer(
        image,
        assets,
        "One learned pupil shapes every depth-dependent exposure",
        "Exact design · fixed global display transform",
        accent=MAGENTA,
    )
    return image


PIPELINE_NODES = (
    (72, 350, 330, 635, "OPTICS T1", "JAX · Chromatix", CYAN),
    (366, 350, 654, 635, "AUTOFOCUS T", "NumPy · SciPy", GREEN),
    (690, 350, 910, 635, "STAGE", "action · µm", MAGENTA),
    (946, 350, 1234, 635, "OPTICS T2", "JAX · Chromatix", CYAN),
    (1270, 350, 1568, 635, "OBSERVER T", "PyTorch · InstanSeg", ORANGE),
    (1604, 350, 1848, 635, "OBJECTIVE", "J", BLUE),
)


def _pipeline_nodes(
    image: Image.Image,
    assets: Assets,
    *,
    active_fraction: float,
    reverse: bool,
) -> None:
    draw = ImageDraw.Draw(image)
    active_step = int(np.clip(active_fraction, 0.0, 0.999) * len(PIPELINE_NODES))
    for index, (x0, y0, x1, y1, title, runtime, color) in enumerate(PIPELINE_NODES):
        if reverse:
            active = index >= len(PIPELINE_NODES) - 1 - active_step
        else:
            active = index <= active_step
        fill = tuple(
            int(PAPER[channel] * 0.25 + color[channel] * 0.05 + 182)
            for channel in range(3)
        )
        fill = tuple(min(255, value) for value in fill) if active else WHITE
        _panel(image, (x0, y0, x1, y1), fill=fill, outline=color if active else LIGHT, width=4)
        _text(
            image,
            ((x0 + x1) // 2, y0 + 74),
            title,
            assets,
            size=30,
            color=color,
            bold=True,
            anchor="ma",
            align="center",
            bounds=(x0 + 10, y0 + 35, x1 - 10, y0 + 118),
        )
        _text(
            image,
            ((x0 + x1) // 2, y0 + 165),
            runtime,
            assets,
            size=27,
            color=NAVY_2,
            anchor="ma",
            align="center",
            bounds=(x0 + 10, y0 + 130, x1 - 10, y0 + 208),
        )
        draw.ellipse(
            (
                (x0 + x1) // 2 - 8,
                y1 - 46,
                (x0 + x1) // 2 + 8,
                y1 - 30,
            ),
            fill=color if active else LIGHT,
        )

    for index, (left, right) in enumerate(
        zip(PIPELINE_NODES, PIPELINE_NODES[1:], strict=False)
    ):
        if reverse:
            reverse_order = len(PIPELINE_NODES) - 2 - index
            fraction = progress(
                active_fraction,
                reverse_order / 5.0,
                (reverse_order + 1) / 5.0,
            )
            _arrow(
                image,
                (right[0] - 12, 492),
                (left[2] + 12, 492),
                color=MAGENTA,
                width=8,
                fraction=fraction,
            )
        else:
            fraction = progress(active_fraction, index / 5.0, (index + 1) / 5.0)
            _arrow(
                image,
                (left[2] + 12, 492),
                (right[0] - 12, 492),
                color=CYAN,
                width=8,
                fraction=fraction,
            )


def scene_forward(local: float, assets: Assets, global_time: float) -> Image.Image:
    image = _base_frame(global_time, assets)
    _header(
        image,
        assets,
        kicker="FORWARD PASS",
        title="One composed closed loop",
        subtitle="signal and stage action move left to right",
        accent=CYAN,
    )
    _pipeline_nodes(image, assets, active_fraction=local / 4.5, reverse=False)
    _text(
        image,
        (960, 728),
        "first exposure  →  predicted focus offset  →  corrected exposure  →  PQ",
        assets,
        size=35,
        color=NAVY_2,
        bold=True,
        anchor="ma",
        bounds=(125, 700, 1795, 775),
    )
    _footer(
        image,
        assets,
        "Three Tesseracts · three native runtimes · one composed workflow.",
        "Optics is called twice: before and after the autofocus stage action.",
        accent=CYAN,
    )
    return image


def scene_reverse(local: float, assets: Assets, global_time: float) -> Image.Image:
    image = _base_frame(global_time, assets)
    _header(
        image,
        assets,
        kicker="EXACT REVERSE GRADIENT",
        title="The objective reaches the pupil parameters",
        accent=MAGENTA,
    )
    reverse_fraction = local / 3.2
    _pipeline_nodes(image, assets, active_fraction=reverse_fraction, reverse=True)
    params_y = 748
    param_width = 112
    gap = 20
    total_width = 7 * param_width + 6 * gap
    start_x = (WIDTH - total_width) // 2
    draw = ImageDraw.Draw(image)
    _arrow(
        image,
        (201, 654),
        (201, 708),
        color=MAGENTA,
        width=7,
        fraction=progress(local, 2.75, 3.25),
    )
    line_fraction = progress(local, 2.7, 3.2)
    if line_fraction > 0:
        draw.line((201, 708, 201 + int(759 * line_fraction), 708), fill=MAGENTA, width=7)
    for index in range(7):
        x0 = start_x + index * (param_width + gap)
        revealed = progress(local, 2.8 + index * 0.08, 3.18 + index * 0.08)
        outline = MAGENTA if revealed > 0.5 else LIGHT
        fill = PALE_MAGENTA if revealed > 0.5 else WHITE
        _panel(
            image,
            (x0, params_y, x0 + param_width, params_y + 82),
            fill=fill,
            outline=outline,
            radius=16,
        )
        _text(
            image,
            (x0 + param_width // 2, params_y + 41),
            f"θ{index + 5}",
            assets,
            size=32,
            color=MAGENTA if revealed > 0.5 else MUTED,
            bold=True,
            anchor="mm",
            bounds=(x0 + 10, params_y + 10, x0 + param_width - 10, params_y + 72),
        )
    _footer(
        image,
        assets,
        "Exact gradient crosses every boundary.",
        "Seven phase parameters optimized end-to-end.",
        accent=MAGENTA,
    )
    return image


def scene_correction(local: float, assets: Assets, global_time: float) -> Image.Image:
    image = _base_frame(global_time, assets)
    _header(
        image,
        assets,
        kicker="CLOSED-LOOP CORRECTION",
        title="See defocus → predict stage move → corrected exposure",
        accent=BLUE,
        title_size=52,
    )
    left = (72, 290, 592, 830)
    middle = (700, 290, 1220, 830)
    right = (1328, 290, 1848, 830)
    for box in (left, middle, right):
        _panel(image, box, fill=WHITE)
    _text(
        image,
        (332, 326),
        "FIRST EXPOSURE",
        assets,
        size=32,
        color=ORANGE,
        bold=True,
        anchor="ma",
        bounds=(95, 300, 570, 360),
    )
    _paste_square(image, assets.before, (142, 375, 522, 755))
    _text(
        image,
        (960, 326),
        "PREDICTED FOCUS OFFSET",
        assets,
        size=31,
        color=MAGENTA,
        bold=True,
        anchor="ma",
        bounds=(720, 300, 1200, 360),
    )
    _text(
        image,
        (960, 495),
        f"{assets.predicted_depth_um:+.2f} µm",
        assets,
        size=78,
        color=MAGENTA,
        bold=True,
        anchor="mm",
        bounds=(730, 420, 1190, 570),
    )
    _arrow(
        image,
        (790, 635),
        (1130, 635),
        color=MAGENTA,
        width=12,
        fraction=progress(local, 1.1, 2.25),
    )
    _text(
        image,
        (960, 715),
        "apply correction",
        assets,
        size=33,
        color=MUTED,
        anchor="ma",
        bounds=(730, 680, 1190, 760),
    )
    _text(
        image,
        (1588, 326),
        "CORRECTED EXPOSURE",
        assets,
        size=32,
        color=BLUE,
        bold=True,
        anchor="ma",
        bounds=(1350, 300, 1825, 360),
    )
    corrected = assets.corrected.copy().convert("RGBA")
    opacity = int(255 * progress(local, 2.1, 3.15))
    corrected.putalpha(opacity)
    _paste_square(image, corrected, (1398, 375, 1778, 755))
    _arrow(
        image,
        (610, 560),
        (682, 560),
        color=ORANGE,
        fraction=progress(local, 0.45, 1.15),
    )
    _arrow(
        image,
        (1238, 560),
        (1310, 560),
        color=BLUE,
        fraction=progress(local, 2.0, 2.8),
    )
    _footer(
        image,
        assets,
        "One matched field, before and after the predicted stage action",
        "Representative −6 µm field · identical display transform",
        accent=BLUE,
    )
    return image


def scene_stage_result(local: float, assets: Assets, global_time: float) -> Image.Image:
    image = _base_frame(global_time, assets)
    _header(
        image,
        assets,
        kicker="DESCRIPTIVE STAGE RESULT",
        title="The closed loop improves hard off-focus frames",
        accent=BLUE,
    )
    _panel(image, (72, 285, 910, 835), fill=WHITE)
    _text(
        image,
        (491, 318),
        "REPRESENTATIVE MATCHED FIELD",
        assets,
        size=30,
        color=MUTED,
        bold=True,
        anchor="ma",
        bounds=(100, 295, 880, 355),
    )
    _paste_square(image, assets.before, (120, 390, 470, 740))
    _paste_square(image, assets.corrected, (512, 390, 862, 740))
    _text(image, (295, 780), "FIRST", assets, size=29, color=ORANGE, bold=True, anchor="ma")
    _text(image, (687, 780), "CORRECTED", assets, size=29, color=BLUE, bold=True, anchor="ma")
    _arrow(
        image,
        (474, 565),
        (506, 565),
        color=BLUE,
        width=7,
        fraction=progress(local, 0.3, 1.0),
    )

    _panel(image, (970, 285, 1848, 835), fill=WHITE, outline=(182, 210, 228))
    animated_after = EXPECTED_CLAIMS["before_pq"] + (
        EXPECTED_CLAIMS["after_pq"] - EXPECTED_CLAIMS["before_pq"]
    ) * progress(local, 0.45, 1.75)
    _text(
        image,
        (1409, 390),
        f"PQ 0.4922 → {animated_after:.4f}",
        assets,
        size=68,
        color=BLUE,
        bold=True,
        anchor="ma",
        bounds=(1000, 340, 1818, 470),
    )
    _text(
        image,
        (1409, 555),
        "74.7%",
        assets,
        size=112,
        color=GREEN,
        bold=True,
        anchor="mm",
        bounds=(1040, 480, 1780, 650),
    )
    _text(
        image,
        (1409, 675),
        "of hard frames improve",
        assets,
        size=43,
        color=NAVY,
        bold=True,
        anchor="ma",
        bounds=(1030, 640, 1790, 725),
    )
    _text(
        image,
        (1409, 767),
        "across the frozen validation population",
        assets,
        size=31,
        color=MUTED,
        anchor="ma",
        bounds=(1010, 735, 1808, 805),
    )
    _footer(
        image,
        assets,
        "PQ 0.4922 → 0.5523",
        "27 hard-density wells · frozen validation",
        accent=BLUE,
    )
    return image


def _value_to_x(value: float) -> int:
    minimum = -0.002
    maximum = 0.013
    return int(round(315 + (value - minimum) / (maximum - minimum) * (1605 - 315)))


def scene_causal(local: float, assets: Assets, global_time: float) -> Image.Image:
    image = _base_frame(global_time, assets)
    _header(
        image,
        assets,
        kicker="CAUSAL GRADIENT EVIDENCE",
        title="Exact reverse gradient adds +0.0063 PQ",
        accent=MAGENTA,
    )
    _panel(image, (150, 285, 1770, 835), fill=WHITE, outline=(221, 199, 218))
    _text(
        image,
        (960, 340),
        "95% CI [+0.0007, +0.0115]",
        assets,
        size=48,
        color=MAGENTA,
        bold=True,
        anchor="ma",
        bounds=(250, 305, 1670, 400),
    )
    draw = ImageDraw.Draw(image)
    axis_y = 590
    draw.line((315, axis_y, 1605, axis_y), fill=NAVY_2, width=5)
    ticks = (-0.002, 0.0, 0.004, 0.008, 0.012)
    for tick in ticks:
        x = _value_to_x(tick)
        draw.line((x, axis_y - 16, x, axis_y + 16), fill=NAVY_2, width=4)
        label = "0" if tick == 0 else f"{tick:+.3f}"
        _text(
            image,
            (x, axis_y + 44),
            label,
            assets,
            size=27,
            color=MUTED,
            anchor="ma",
        )
    zero_x = _value_to_x(0.0)
    draw.line((zero_x, 452, zero_x, 700), fill=LIGHT, width=4)
    ci_low = _value_to_x(EXPECTED_CLAIMS["causal_ci_lower_95"])
    ci_high = _value_to_x(EXPECTED_CLAIMS["causal_ci_upper_95"])
    point = _value_to_x(EXPECTED_CLAIMS["causal_gain"])
    ci_fraction = progress(local, 0.25, 1.1)
    animated_high = int(ci_low + (ci_high - ci_low) * ci_fraction)
    draw.line((ci_low, axis_y, animated_high, axis_y), fill=MAGENTA, width=18)
    if ci_fraction > 0.96:
        draw.line((ci_low, axis_y - 34, ci_low, axis_y + 34), fill=MAGENTA, width=7)
        draw.line((ci_high, axis_y - 34, ci_high, axis_y + 34), fill=MAGENTA, width=7)
        draw.ellipse((point - 19, axis_y - 19, point + 19, axis_y + 19), fill=BLUE)
    _text(
        image,
        (960, 720),
        "Forward-identical stopped control",
        assets,
        size=42,
        color=NAVY,
        bold=True,
        anchor="ma",
        bounds=(300, 680, 1620, 770),
    )
    _text(
        image,
        (960, 785),
        "paired mean difference in panoptic quality (PQ; 0–1)",
        assets,
        size=30,
        color=MUTED,
        anchor="ma",
        bounds=(260, 755, 1660, 825),
    )
    _footer(
        image,
        assets,
        "Causal evidence from the exact gradient path",
        "27 hard-density wells · 2,000 grouped bootstrap replicates",
        accent=MAGENTA,
    )
    return image


def scene_lockup(local: float, assets: Assets, global_time: float) -> Image.Image:
    image = _base_frame(global_time, assets)
    ImageDraw.Draw(image).rectangle((0, 0, 440, 80), fill=WHITE)
    _text(
        image,
        (960, 100),
        "TessScope",
        assets,
        size=104,
        color=NAVY,
        bold=True,
        anchor="ma",
        bounds=(400, 90, 1520, 225),
    )
    _text(
        image,
        (960, 245),
        "Differentiable closed-loop microscopy",
        assets,
        size=48,
        color=BLUE,
        bold=True,
        anchor="ma",
        bounds=(310, 215, 1610, 300),
    )
    cards = (
        (486, assets.object_image, "INPUT", BLUE),
        (826, assets.corrected, "CORRECTION", ORANGE),
        (1166, assets.pupil, "OPTIMIZED\nPUPIL", MAGENTA),
    )
    for x, source, label, color in cards:
        _panel(image, (x, 345, x + 268, 690), fill=WHITE, outline=color, width=4)
        _paste_square(image, source, (x + 24, 370, x + 244, 590))
        _text(
            image,
            (x + 134, 640),
            label,
            assets,
            size=29,
            color=color,
            bold=True,
            anchor="mm",
            align="center",
            spacing=2,
            bounds=(x + 10, 605, x + 258, 675),
        )
    pulse = 0.65 + 0.35 * math.sin(local * math.pi * 1.3) ** 2
    line_color = tuple(int(value * pulse) for value in CYAN)
    _arrow(image, (760, 515), (816, 515), color=line_color, width=8)
    _arrow(image, (1100, 515), (1156, 515), color=line_color, width=8)
    _text(
        image,
        (960, 760),
        "INPUT → ACTION → CORRECTION → EXACT GRADIENT → OPTIMIZED PUPIL",
        assets,
        size=34,
        color=NAVY_2,
        bold=True,
        anchor="ma",
        bounds=(145, 730, 1775, 805),
    )
    _footer(
        image,
        assets,
        "Built to make the whole differentiable loop inspectable",
        "Hackathon demo · frozen validation · cached replay",
        accent=CYAN,
    )
    return image


SCENE_RENDERERS = (
    scene_input,
    scene_optics,
    scene_forward,
    scene_reverse,
    scene_correction,
    scene_stage_result,
    scene_causal,
    scene_lockup,
)


def render_frame(time_seconds: float, assets: Assets) -> Image.Image:
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
        blend = progress(time_seconds, end - TRANSITION_SECONDS, end)
        following = SCENE_RENDERERS[index + 1](blend * TRANSITION_SECONDS, assets, time_seconds)
        split = int(round(WIDTH * blend))
        if split > 0:
            current.alpha_composite(following.crop((0, 0, split, HEIGHT)), (0, 0))
            ImageDraw.Draw(current).line((split, 0, split, HEIGHT), fill=CYAN, width=8)
    return current.convert("RGB")


def project_path(value: Path) -> Path:
    path = value if value.is_absolute() else PROJECT_ROOT / value
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Demo output must stay inside the project directory")
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
        "title=TessScope — 29-second end-to-end demo",
        "-metadata",
        "comment=Frozen validation cached replay; intentional no-audio stream",
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
            "-count_frames",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames",
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


def extract_frame(ffmpeg: str, video: Path, frame_index: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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
            str(destination),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _stamp(image: Image.Image, timestamp: float, assets: Assets, *, small: bool) -> None:
    draw = ImageDraw.Draw(image)
    if small:
        box = (505, 320, 632, 354)
        selected_font = _font(assets, 18, bold=True)
        text_xy = (620, 337)
    else:
        box = (1515, 960, 1895, 1052)
        selected_font = _font(assets, 42, bold=True)
        text_xy = (1865, 1006)
    draw.rounded_rectangle(box, radius=12 if small else 24, fill=(8, 24, 45))
    draw.text(text_xy, f"{timestamp:05.2f}s", font=selected_font, fill=WHITE, anchor="rm")


def create_review_assets(ffmpeg: str, output: Path, assets: Assets) -> list[dict]:
    review_root = PROJECT_ROOT / ".release-check" / "end-to-end-demo-frames"
    review_root.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    records: list[dict] = []
    for index, timestamp in enumerate(SAMPLE_TIMESTAMPS):
        frame_index = min(FRAME_COUNT - 1, round(timestamp * FPS))
        frame_path = review_root / f"frame-{index + 1:02d}-{frame_index:04d}.png"
        extract_frame(ffmpeg, output, frame_index, frame_path)
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB"))
        records.append(
            {
                "timestamp_seconds": timestamp,
                "frame_index": frame_index,
                "encoded_frame_sha256": sha256_path(frame_path),
            }
        )

    full_sheet = Image.new("RGB", (WIDTH * 3, HEIGHT * 3), PAPER)
    readability = Image.new("RGB", (640 * 3, 360 * 3), PAPER)
    for index, (timestamp, frame) in enumerate(zip(SAMPLE_TIMESTAMPS, frames, strict=True)):
        row, column = divmod(index, 3)
        full = frame.copy()
        _stamp(full, timestamp, assets, small=False)
        full_sheet.paste(full, (column * WIDTH, row * HEIGHT))

        small_frame = frame.resize((640, 360), Image.Resampling.LANCZOS)
        _stamp(small_frame, timestamp, assets, small=True)
        readability.paste(small_frame, (column * 640, row * 360))

    CONTACT_SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_sheet.save(CONTACT_SHEET_PATH, optimize=False)
    readability.save(READABILITY_SHEET_PATH, optimize=False)
    poster_frame = round(22.5 * FPS)
    extract_frame(ffmpeg, output, poster_frame, POSTER_PATH)
    return records


def verify_claim_sources(assets: Assets) -> dict:
    stage = assets.validation["stage_correction"][DESIGN_NAME]
    causal = assets.validation["paired_evidence"][DESIGN_NAME]["corrected_vs_stopped"]
    actual = {
        "before_pq": stage["hard_off_focus_pq_before"],
        "after_pq": stage["hard_off_focus_pq_after"],
        "improved_fraction": stage["fraction_hard_frames_improved"],
        "causal_gain": causal["mean_difference"],
        "causal_ci_lower_95": causal["ci_lower_95"],
        "causal_ci_upper_95": causal["ci_upper_95"],
        "source_count": causal["source_count"],
        "bootstrap_replicates": causal["replicates"],
    }
    for key, expected in EXPECTED_CLAIMS.items():
        value = actual[key]
        if isinstance(expected, float):
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"Frozen end-to-end claim changed: {key}")
        elif value != expected:
            raise ValueError(f"Frozen end-to-end claim changed: {key}")
    return actual


def preserved_video_hashes() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in PRESERVED_VIDEO_PATHS:
        key = str(path.relative_to(PROJECT_ROOT))
        records[key] = {
            "exists": path.is_file(),
            "sha256": sha256_path(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return records


def write_manifest(
    output: Path,
    assets: Assets,
    frame_records: list[dict],
    preserved: dict[str, dict],
    ffprobe_metadata: dict,
) -> None:
    claims = verify_claim_sources(assets)
    selected_patch = next(
        patch for patch in assets.metadata["patches"] if patch["field_id"] == "n21_s1"
    )
    source_paths = (
        REPLAY_PATH,
        REPLAY_METADATA_PATH,
        VALIDATION_PATH,
        DERIVATIVE_PATH,
        VISUAL_CONTRACT_PATH,
        PROJECT_ROOT / selected_patch["object_source"],
    )
    manifest = {
        "schema_version": 1,
        "status": "complete_review_candidate",
        "selected_candidate": False,
        "external_actions_performed": [],
        "media_policy": {
            "scientific_media": "frozen traced TessScope assets only",
            "synthetic_or_ai_scientific_media": False,
            "per_image_normalization": False,
            "sharpening": False,
            "denoising": False,
            "geometry_changes_between_compared_panels": False,
            "audio": "intentional no-audio stream; muted-first edit",
        },
        "claim_boundary": {
            "scope": "frozen validation",
            "test_accessed": False,
            "live_inference": False,
            "physical_microscope_validation_claimed": False,
            "state_of_the_art_claimed": False,
            "causal_wording": "causal gradient evidence; not causal proof",
        },
        "format": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "duration_seconds": DURATION_SECONDS,
            "frame_count": FRAME_COUNT,
            "codec": "H.264",
            "pixel_format": "yuv420p",
            "fast_start": True,
            "ffprobe": ffprobe_metadata,
        },
        "timeline": [
            {"id": name, "start_seconds": start, "end_seconds": end}
            for name, start, end in SCENES
        ],
        "claims": {
            "descriptive_stage_result": {
                "display": {
                    "pq": "PQ 0.4922 → 0.5523",
                    "improvement": "74.7% of hard frames improve",
                },
                "raw": {
                    "before_pq": claims["before_pq"],
                    "after_pq": claims["after_pq"],
                    "improved_fraction": claims["improved_fraction"],
                },
                "source": str(VALIDATION_PATH.relative_to(PROJECT_ROOT)),
                "pointers": [
                    f"/stage_correction/{DESIGN_NAME}/hard_off_focus_pq_before",
                    f"/stage_correction/{DESIGN_NAME}/hard_off_focus_pq_after",
                    f"/stage_correction/{DESIGN_NAME}/fraction_hard_frames_improved",
                ],
            },
            "causal_gradient_evidence": {
                "display": {
                    "gain": "+0.0063 PQ",
                    "interval": "95% CI [+0.0007, +0.0115]",
                    "control": "Forward-identical stopped control",
                },
                "raw": {
                    "mean_difference": claims["causal_gain"],
                    "ci_lower_95": claims["causal_ci_lower_95"],
                    "ci_upper_95": claims["causal_ci_upper_95"],
                    "source_count": claims["source_count"],
                    "replicates": claims["bootstrap_replicates"],
                },
                "source": str(VALIDATION_PATH.relative_to(PROJECT_ROOT)),
                "pointer": f"/paired_evidence/{DESIGN_NAME}/corrected_vs_stopped",
            },
        },
        "scientific_sources": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(path),
            }
            for path in source_paths
        ],
        "scientific_transforms": {
            "input_object": (
                "clip-only display of the already globally normalized replay "
                "object_image array"
            ),
            "sensor_panels": assets.metadata["observer"]["transform"],
            "segmentation_boundaries": (
                "tessscope.demo.figures.overlay_boundaries; identical scale "
                "and geometry"
            ),
            "pupil": "wrapped exact-design phase rendered with fixed twilight colormap",
            "psf": "global replay maximum, fixed log10 floor 1e-5, fixed magma colormap",
        },
        "preserved_videos": preserved,
        "outputs": {
            "video": {
                "path": str(output.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(output),
                "bytes": output.stat().st_size,
            },
            "poster": {
                "path": str(POSTER_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(POSTER_PATH),
                "dimensions": list(Image.open(POSTER_PATH).size),
            },
            "full_resolution_contact_sheet": {
                "path": str(CONTACT_SHEET_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(CONTACT_SHEET_PATH),
                "dimensions": list(Image.open(CONTACT_SHEET_PATH).size),
                "tile_dimensions": [WIDTH, HEIGHT],
            },
            "readability_contact_sheet": {
                "path": str(READABILITY_SHEET_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(READABILITY_SHEET_PATH),
                "dimensions": list(Image.open(READABILITY_SHEET_PATH).size),
                "tile_dimensions": [640, 360],
            },
            "storyboard": {
                "path": str(STORYBOARD_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(STORYBOARD_PATH),
            },
            "on_screen_copy": {
                "path": str(COPY_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(COPY_PATH),
            },
        },
        "encoded_frame_review_samples": frame_records,
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
    assets = load_assets()
    encode_video(ffmpeg, output, assets)
    preserved_after = preserved_video_hashes()
    if preserved_before != preserved_after:
        raise ValueError("A preserved earlier video changed during the build")

    metadata = probe(ffprobe, output)
    if output == DEFAULT_OUTPUT.resolve():
        frame_records = create_review_assets(ffmpeg, output, assets)
        write_manifest(output, assets, frame_records, preserved_after, metadata)
        print(
            json.dumps(
                {
                    "status": "rendered_review_candidate",
                    "output": str(output.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_path(output),
                    "frames": FRAME_COUNT,
                    "review_assets": len(frame_records),
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "status": "rendered_noncanonical_comparison",
                    "output": str(output.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_path(output),
                    "ffprobe": metadata,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
