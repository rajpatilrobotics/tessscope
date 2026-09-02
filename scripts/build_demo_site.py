"""Build deterministic data and traceability for the dependency-free judge site."""

from __future__ import annotations

import json
from pathlib import Path

from tessscope.demo.evidence import sha256_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "demo"
REPLAY_METADATA_PATH = (
    PROJECT_ROOT / "artifacts" / "runs" / "demo" / "validation-replay.json"
)
FIGURE_MANIFEST_PATH = OUTPUT_ROOT / "figure-manifest.json"
SITE_DATA_PATH = OUTPUT_ROOT / "site-data.json"
SITE_MANIFEST_PATH = OUTPUT_ROOT / "site-manifest.json"
SITE_FILES = ("index.html", "site.css", "app.js", "site-data.json")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _frame(metadata: dict, system: str, depth: float) -> dict:
    matches = [
        row
        for row in metadata["frames"]
        if row["system"] == system and row["depth_um"] == depth
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {system} frame at {depth}")
    return matches[0]


def _system_payload(metadata: dict, system: str, depth: float) -> dict:
    trace = _frame(metadata, system, depth)
    return {
        "id": system,
        "label": {
            "exact": "Exact gradient",
            "stopped": "Stopped gradient",
            "piecewise": "Piecewise-028",
        }[system],
        "before_pq": trace["before_pq"],
        "after_pq": trace.get("after_pq"),
        "predicted_depth_um": trace["predicted_depth_um"],
        "residual_depth_um": trace["residual_depth_um"],
        "source": trace["source"],
        "source_sha256": trace["source_sha256"],
        "before_pointer": trace["before_pointer"],
        "corrected_pointer": trace.get("corrected_pointer"),
    }


def main() -> None:
    metadata = _json(REPLAY_METADATA_PATH)
    figures = _json(FIGURE_MANIFEST_PATH)
    if metadata["split"] != "validation" or metadata["test_accessed"]:
        raise ValueError("Judge site accepts only the sealed validation replay")
    if figures["test_accessed"]:
        raise ValueError("Judge site accepts only sealed figure outputs")

    frames = []
    for index, depth in enumerate(metadata["depths_um"]):
        systems = [
            _system_payload(metadata, system, depth)
            for system in metadata["corrected_system_order"]
        ]
        depth_text = "0 µm" if depth == 0 else f"{'+' if depth > 0 else '−'}{abs(depth):.0f} µm"
        frames.append(
            {
                "index": index,
                "depth_um": depth,
                "path": f"replay/frame-{index:02d}.png",
                "path_sha256": figures["replay"]["frames"][index]["sha256"],
                "representative_still": depth == metadata["selected_display_depth_um"],
                "caption": (
                    f"Frozen representative field {metadata['selected_field']} at "
                    f"{depth_text} initial defocus."
                ),
                "alt": (
                    f"Cached validation replay at {depth_text} comparing exact, "
                    "stopped-gradient, and piecewise first and corrected frames."
                ),
                "systems": systems,
            }
        )
    site_data = {
        "schema_version": 1,
        "status": "cached_validation_replay_not_live",
        "test_accessed": False,
        "selected_field": metadata["selected_field"],
        "selected_display_depth_um": metadata["selected_display_depth_um"],
        "frames": frames,
        "headline": {
            "exact_vs_stopped_pq": 0.006258840610359453,
            "ci_lower_95": 0.0006703181908061804,
            "ci_upper_95": 0.011547458540670894,
            "exact_vs_piecewise_pq": 0.0012850136904830492,
            "piecewise_ci_lower_95": -0.001719244321047897,
            "piecewise_ci_upper_95": 0.004800378815350437,
        },
        "limitations": figures["limitations"],
        "inputs": {
            "replay_metadata": {
                "path": str(REPLAY_METADATA_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(REPLAY_METADATA_PATH),
            },
            "figure_manifest": {
                "path": str(FIGURE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(FIGURE_MANIFEST_PATH),
            },
        },
    }
    SITE_DATA_PATH.write_text(json.dumps(site_data, indent=2, sort_keys=True) + "\n")

    site_files = {}
    for name in SITE_FILES:
        path = OUTPUT_ROOT / name
        site_files[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(path),
            "size_bytes": path.stat().st_size,
        }
    assets = {}
    for figure in figures["figures"].values():
        png = figure["files"]["png"]
        assets[png["path"]] = {"sha256": png["sha256"], "role": "publication_figure"}
    for frame in figures["replay"]["frames"]:
        assets[frame["path"]] = {"sha256": frame["sha256"], "role": "depth_replay_frame"}
    for format_name in ("gif", "mp4"):
        output = figures["replay"][format_name]
        assets[output["path"]] = {"sha256": output["sha256"], "role": "depth_animation"}
    manifest = {
        "schema_version": 1,
        "status": "complete_local_static_judge_demo",
        "mode": "cached_replay_not_live_inference",
        "test_accessed": False,
        "entrypoint": "python scripts/serve_demo.py",
        "network_binding": "127.0.0.1",
        "deployment": "not_performed",
        "site_files": site_files,
        "assets": dict(sorted(assets.items())),
        "inputs": site_data["inputs"],
    }
    SITE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"frames={len(frames)}")
    print(f"site_data_sha256={sha256_path(SITE_DATA_PATH)}")
    print(f"site_manifest_sha256={sha256_path(SITE_MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
