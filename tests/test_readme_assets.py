from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
readme_assets = importlib.import_module("scripts.build_readme_assets")
OUTPUT_ROOT = readme_assets.OUTPUT_ROOT
build_assets = readme_assets.build_assets


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readme_assets_are_source_traceable_and_deterministic() -> None:
    first = build_assets()
    first_hashes = {name: value["sha256"] for name, value in first["outputs"].items()}
    first_hashes.update(
        {name: value["sha256"] for name, value in first["animations"].items()}
    )
    second = build_assets()
    second_hashes = {name: value["sha256"] for name, value in second["outputs"].items()}
    second_hashes.update(
        {name: value["sha256"] for name, value in second["animations"].items()}
    )

    assert first_hashes == second_hashes
    assert first["scientific_integrity"]["generated_scientific_imagery"] is False
    for relative, expected_hash in first["sources"].items():
        assert _sha256(PROJECT_ROOT / relative) == expected_hash


def test_readme_asset_dimensions_and_manifest() -> None:
    manifest = build_assets()
    expected_sizes = {
        "video_poster": (1600, 900),
        "problem": (1600, 900),
        "solution": (1600, 900),
        "system_proof": (1600, 900),
        "gradient_path": (1800, 900),
    }
    for name, expected in expected_sizes.items():
        path = PROJECT_ROOT / manifest["outputs"][name]["path"]
        with Image.open(path) as image:
            assert image.size == expected

    on_disk = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 2
    assert set(on_disk["outputs"]) == set(expected_sizes)


def test_readme_animated_assets_are_looping_and_bounded() -> None:
    manifest = build_assets()
    expected_names = {
        "video_preview_gif",
        "problem_gif",
        "solution_gif",
        "graph_gif",
        "system_proof_gif",
    }
    assert set(manifest["animations"]) == expected_names

    total_bytes = 0
    for details in manifest["animations"].values():
        path = PROJECT_ROOT / details["path"]
        with Image.open(path) as image:
            assert image.size == (960, 540)
            assert image.n_frames > 1
            assert image.info.get("loop") == 0
        assert details["duration_seconds"] > 3
        assert details["source_duration_seconds"] > 3
        assert details["bytes"] == path.stat().st_size
        assert path.stat().st_size < 2_000_000
        total_bytes += path.stat().st_size

    assert total_bytes < 8_000_000
    assert manifest["scientific_integrity"]["generated_scientific_imagery"] is False


def test_readme_uses_the_final_demo_and_only_existing_local_links() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://youtu.be/u5QRV6fVLQg" in readme
    assert "tessscope-end-to-end-demo-polished-v23.mp4" in readme
    for name in ("video-preview", "problem", "solution", "system-proof"):
        assert f"outputs/demo/readme/{name}.gif" in readme
    assert "outputs/demo/readme/graph.gif" in readme
    assert "outputs/demo/readme/gradient-path.png" in readme

    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
    local_targets = [link.split("#", 1)[0] for link in links if not link.startswith("http")]
    assert local_targets
    assert all((PROJECT_ROOT / target).exists() for target in local_targets)


def test_readme_keeps_protected_claims_and_limitations_together() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    for required in (
        "+0.006259",
        "+0.000670",
        "+0.011547",
        "27 hard-density validation wells",
        "2,000 deterministic paired bootstrap replicates",
        "PQ is panoptic quality",
        "locked BBBC006 test split remains sealed",
        "no physical microscope has validated the system",
        "Superiority is not supported",
    ):
        assert required in normalized_readme


def test_readme_states_scientific_community_and_tesseract_case() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    for required in (
        "Scientific impact and community extension paths",
        "Computational microscopy and optical design",
        "Bioimage analysis and high-content screening",
        "Microscopy laboratories and instrument automation",
        "This integration is future work, not a current claim.",
        "Why Tesseract, rather than ordinary glue code?",
        "rewrite specialized components into one automatic-differentiation stack",
        "End-to-end optimization",
        "Research reuse",
    ):
        assert required in normalized_readme


def test_readme_uses_github_safe_math_fences() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.count("```math") == 6
    assert "$$" not in readme
    assert r"\(" not in readme
    assert r"\operatorname" not in readme
