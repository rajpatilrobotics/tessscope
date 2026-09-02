"""Assemble the final end-to-end traceability manifest for every demo output."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tessscope.demo.evidence import sha256_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "demo"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "runs" / "demo"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "demo"
CLAIMS_PATH = CONFIG_ROOT / "claim-matrix.yaml"
VISUAL_PATH = CONFIG_ROOT / "visual-contract.yaml"
SOURCES_PATH = CONFIG_ROOT / "source-manifest.json"
SELECTION_PATH = ARTIFACT_ROOT / "representative-selection.json"
REPLAY_METADATA_PATH = ARTIFACT_ROOT / "validation-replay.json"
REPLAY_ARRAYS_PATH = ARTIFACT_ROOT / "validation-replay.npz"
FIGURE_MANIFEST_PATH = OUTPUT_ROOT / "figure-manifest.json"
SITE_DATA_PATH = OUTPUT_ROOT / "site-data.json"
SITE_MANIFEST_PATH = OUTPUT_ROOT / "site-manifest.json"
OUTPUT_PATH = OUTPUT_ROOT / "traceability-manifest.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _pointer(document: object, pointer: str) -> object:
    value = document
    for raw_token in pointer.removeprefix("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _claim_trace(claim: dict) -> dict:
    source_path = PROJECT_ROOT / claim["source"]
    document = _json(source_path)
    return {
        "id": claim["id"],
        "text": claim["text"],
        "status": claim["status"],
        "source": claim["source"],
        "source_sha256": sha256_path(source_path),
        "values": {
            name: {"pointer": pointer, "raw": _pointer(document, pointer)}
            for name, pointer in claim.get("pointers", {}).items()
        },
        "display": claim.get("display", {}),
    }


def _add_output(inventory: dict, path_text: str, role: str) -> None:
    path = PROJECT_ROOT / path_text
    if not path.is_file():
        raise FileNotFoundError(f"Missing traceability output: {path_text}")
    current = inventory.get(path_text)
    if current is not None and current["role"] != role:
        current["role"] = f"{current['role']};{role}"
        return
    inventory[path_text] = {
        "role": role,
        "sha256": sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _output_inventory(figures: dict, site: dict) -> dict:
    inventory: dict[str, dict] = {}
    for figure_id, figure in figures["figures"].items():
        for suffix, output in figure["files"].items():
            _add_output(inventory, output["path"], f"figure:{figure_id}:{suffix}")
    for frame in figures["replay"]["frames"]:
        _add_output(inventory, frame["path"], f"depth_replay_frame:{frame['depth_um']:+.1f}")
    for format_name in ("gif", "mp4"):
        _add_output(
            inventory,
            figures["replay"][format_name]["path"],
            f"depth_replay_animation:{format_name}",
        )
    _add_output(inventory, figures["captions"]["path"], "captions_and_alt_text")
    for name, output in site["site_files"].items():
        _add_output(inventory, output["path"], f"judge_site:{name}")
    for path, role in (
        (FIGURE_MANIFEST_PATH, "figure_manifest"),
        (SITE_MANIFEST_PATH, "site_manifest"),
        (SELECTION_PATH, "representative_selection"),
        (REPLAY_METADATA_PATH, "validation_replay_metadata"),
        (REPLAY_ARRAYS_PATH, "validation_replay_arrays"),
        (CLAIMS_PATH, "claim_contract"),
        (VISUAL_PATH, "visual_contract"),
        (SOURCES_PATH, "source_manifest"),
    ):
        _add_output(inventory, str(path.relative_to(PROJECT_ROOT)), role)
    return dict(sorted(inventory.items()))


def main() -> None:
    claims = yaml.safe_load(CLAIMS_PATH.read_text())
    visual = yaml.safe_load(VISUAL_PATH.read_text())
    sources = _json(SOURCES_PATH)
    selection = _json(SELECTION_PATH)
    replay = _json(REPLAY_METADATA_PATH)
    figures = _json(FIGURE_MANIFEST_PATH)
    site_data = _json(SITE_DATA_PATH)
    site = _json(SITE_MANIFEST_PATH)
    if any(
        payload["test_accessed"]
        for payload in (sources["data_boundary"], selection, replay, figures, site_data, site)
    ):
        raise ValueError("Cannot assemble demo traceability after locked-test access")

    traced_claims = [_claim_trace(claims["protected_claim"])]
    traced_claims.extend(_claim_trace(claim) for claim in claims["claims"])
    inventory = _output_inventory(figures, site)
    archive_size = replay["archive"]["size_bytes"]
    manifest = {
        "schema_version": 1,
        "status": "complete_end_to_end_validation_traceability",
        "test_accessed": False,
        "source_commit_before_demo_phase": sources["source_commit"],
        "scope": {
            "evaluation_split": "validation",
            "allowed_splits": visual["data_boundary"]["allowed_splits"],
            "forbidden_split": visual["data_boundary"]["forbidden_split"],
            "demo_mode": site["mode"],
            "deployment": site["deployment"],
        },
        "claim_trace": traced_claims,
        "required_limitations": claims["limitations"],
        "representative_sample": {
            "selection_path": str(SELECTION_PATH.relative_to(PROJECT_ROOT)),
            "selection_sha256": sha256_path(SELECTION_PATH),
            "rule": selection["selection_rule"],
            "population": selection["population"],
            "selected_patch": selection["selected_patch"],
            "selected_field_metrics": selection["selected_field"],
            "selected_display_depth": selection["selected_display_depth"],
        },
        "replay_arrays": {
            "archive": replay["archive"],
            "metadata_path": str(REPLAY_METADATA_PATH.relative_to(PROJECT_ROOT)),
            "metadata_sha256": sha256_path(REPLAY_METADATA_PATH),
            "array_entries": replay["arrays"],
            "patch_sources": replay["patches"],
            "design_sources": replay["designs"],
            "display_transform": replay["observer"]["transform"],
            "calibration": replay["calibration"],
        },
        "displayed_frame_values": site_data["frames"],
        "displayed_population_curves": {
            "pq_curves": figures["figures"]["pq-focus-depth"]["pq_curves"],
            "focus_curves": figures["figures"]["pq-focus-depth"]["focus_curves"],
            "source_rows": figures["figures"]["pq-focus-depth"]["source_rows"],
            "derivation": (
                "For each design and source depth, take the 27 hard-field well values; "
                "report their mean and a deterministic 2,000-replicate percentile bootstrap "
                "that resamples wells."
            ),
        },
        "displayed_optical_values": {
            "array_entries": {
                key: replay["arrays"][key]
                for key in ("phase_parameters", "pupil_phase_radians", "pupil_mask", "psf_sensor")
            },
            "render": figures["figures"]["pupil-psf-depth"]["psf_display"],
        },
        "displayed_architecture_values": figures["figures"]["gradient-architecture"][
            "derivative_gate"
        ],
        "demo_static_facts": {
            "source_pack_size_bytes": archive_size,
            "source_pack_size_display_mib": round(archive_size / (1024 * 1024), 1),
            "depth_count": len(replay["depths_um"]),
            "system_count": len(replay["system_order"]),
            "source": str(REPLAY_METADATA_PATH.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_path(REPLAY_METADATA_PATH),
        },
        "manifest_chain": {
            "source_manifest": {
                "path": str(SOURCES_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(SOURCES_PATH),
            },
            "figure_manifest": {
                "path": str(FIGURE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(FIGURE_MANIFEST_PATH),
            },
            "site_manifest": {
                "path": str(SITE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(SITE_MANIFEST_PATH),
            },
        },
        "output_inventory": inventory,
    }
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"claims={len(traced_claims)}")
    print(f"outputs={len(inventory)}")
    print(f"traceability_sha256={sha256_path(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
