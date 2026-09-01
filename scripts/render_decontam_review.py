"""Render identity-only contact sheets for flagged BBBC039/BBBC038 pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tessscope.data.decontam import (
    BBBC038_ROOT,
    BBBC039_ROOT,
    canonical_gray,
    d4_views,
    normalized_view,
)


def _display(view: np.ndarray, size: int = 160) -> Image.Image:
    low, high = np.percentile(view, [0.5, 99.5])
    scaled = np.clip((view - low) / max(1e-8, high - low), 0, 1)
    image = Image.fromarray(np.round(scaled * 255).astype(np.uint8), mode="L")
    return image.resize((size, size), Image.Resampling.NEAREST)


def main() -> None:
    manifest = Path("data/manifests/bbbc-decontamination.json")
    report = json.loads(manifest.read_text())
    candidates = report["candidates"]
    output_dir = Path("artifacts/runs/data/decontamination-review")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_per_page = 10
    panel_size = 160
    row_height = 205
    width = panel_size * 2 + 40
    for page_index, offset in enumerate(range(0, len(candidates), rows_per_page), start=1):
        page_candidates = candidates[offset : offset + rows_per_page]
        canvas = Image.new("RGB", (width, 30 + row_height * len(page_candidates)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), "BBBC039 (D4 aligned)      BBBC038 — identity review only", fill="black")
        for row, candidate in enumerate(page_candidates):
            top = 30 + row * row_height
            path_039 = BBBC039_ROOT / candidate["bbbc039_path"]
            path_038 = BBBC038_ROOT / candidate["bbbc038_path"]
            view_039 = d4_views(normalized_view(canonical_gray(path_039)))[
                candidate["bbbc039_d4_orientation"]
            ]
            view_038 = normalized_view(canonical_gray(path_038))
            canvas.paste(_display(view_039), (10, top))
            canvas.paste(_display(view_038), (panel_size + 30, top))
            text = (
                f"#{offset + row + 1} {candidate['bbbc039_split']} "
                f"NCC={candidate['maximum_d4_ncc']:.6f} pHash={candidate['minimum_phash_hamming']}"
            )
            draw.text((10, top + panel_size + 5), text, fill="black")
            draw.text((10, top + panel_size + 20), candidate["bbbc039_id"][:45], fill="black")
        canvas.save(output_dir / f"review-{page_index:02d}.png")
    print(f"Wrote {(len(candidates) + rows_per_page - 1) // rows_per_page} review sheets")


if __name__ == "__main__":
    main()
