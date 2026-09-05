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
    second = build_assets()
    second_hashes = {name: value["sha256"] for name, value in second["outputs"].items()}

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
        assert Image.open(path).size == expected

    on_disk = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 1
    assert set(on_disk["outputs"]) == set(expected_sizes)


def test_readme_uses_the_final_demo_and_only_existing_local_links() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://youtu.be/u5QRV6fVLQg" in readme
    assert "tessscope-end-to-end-demo-polished-v23.mp4" in readme
    for name in ("video-poster", "problem", "solution", "system-proof", "gradient-path"):
        assert f"outputs/demo/readme/{name}.png" in readme

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


def test_readme_uses_github_safe_math_fences() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.count("```math") == 6
    assert "$$" not in readme
    assert r"\(" not in readme
    assert r"\operatorname" not in readme
