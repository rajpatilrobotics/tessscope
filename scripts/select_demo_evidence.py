"""Select the frozen representative validation field before visual rendering."""

from __future__ import annotations

import json
from pathlib import Path

from tessscope.demo.evidence import select_representative, sha256_path
from tessscope.v2.optimization.served import collect_patches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = PROJECT_ROOT / "configs" / "demo" / "claim-matrix.yaml"
VISUAL = PROJECT_ROOT / "configs" / "demo" / "visual-contract.yaml"
SOURCES = PROJECT_ROOT / "configs" / "demo" / "source-manifest.json"
V2_4 = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_4"
    / "validation"
    / "expanded-hard-validation.json"
)
PIECEWISE = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "v2_2"
    / "validation"
    / "piecewise-hard-frontier.json"
)
OUTPUT = PROJECT_ROOT / "artifacts" / "runs" / "demo" / "representative-selection.json"


def main() -> None:
    selection = select_representative(
        json.loads(V2_4.read_text()), json.loads(PIECEWISE.read_text())
    )
    selected_field = selection["selected_field"]["field_id"]
    patches = collect_patches("validation", 45, seed=53)
    matches = [patch for patch in patches if patch.field_id == selected_field]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen validation patch for {selected_field}")
    patch = matches[0]
    selection.update(
        {
            "status": "selected_before_visual_rendering",
            "test_accessed": False,
            "split": patch.split,
            "selected_patch": {
                "field_id": patch.field_id,
                "well": patch.well,
                "site": patch.site,
                "origin_yx": list(patch.origin_yx),
                "shape_px": list(patch.object_image.shape),
                "valid_instance_count": int(patch.valid_objects.sum()),
            },
            "frozen_inputs": {
                "claim_matrix_sha256": sha256_path(CLAIMS),
                "visual_contract_sha256": sha256_path(VISUAL),
                "source_manifest_sha256": sha256_path(SOURCES),
                "v2_4_hard_validation_sha256": sha256_path(V2_4),
                "piecewise_hard_validation_sha256": sha256_path(PIECEWISE),
            },
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"selected_field={selected_field}")
    print(f"selected_display_depth_um={selection['selected_display_depth']['depth_um']}")
    print(f"output={OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"sha256={sha256_path(OUTPUT)}")


if __name__ == "__main__":
    main()
