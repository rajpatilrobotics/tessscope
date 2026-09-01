"""Record the immutable pre-v2 TessScope repository and evidence snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "v1" / "freeze.json"

INCLUDED_ROOTS = (
    "src",
    "services",
    "scripts",
    "tests",
    "configs",
    "data/manifests",
    "outputs",
    "artifacts/runs",
)
INCLUDED_FILES = (
    ".gitignore",
    ".python-version",
    "NOTICE.md",
    "PROJECT_BRIEF.md",
    "README.md",
    "decision-log.md",
    "plan.md",
    "pyproject.toml",
    "uv.lock",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_paths() -> list[Path]:
    paths = [PROJECT_ROOT / name for name in INCLUDED_FILES]
    for root_name in INCLUDED_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(
        {
            path
            for path in paths
            if path.exists()
            and "__pycache__" not in path.parts
            and not path.name.endswith(".pyc")
            and path != OUTPUT_PATH
            and path.name != "v2_freeze_v1.py"
        }
    )


def main() -> None:
    if OUTPUT_PATH.exists():
        raise SystemExit(
            "The v1 freeze already exists. Refusing to overwrite immutable evidence."
        )
    files = {
        str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in included_paths()
    }
    payload = {
        "experiment": "tessscope_v1",
        "status": "frozen_before_v2_implementation",
        "frozen_at_local_date": "2026-09-01",
        "policy": (
            "V2 uses separately named modules, configs, artifacts, and scripts. "
            "A changed hash must never be described as the original v1 experiment."
        ),
        "file_count": len(files),
        "sha256": files,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT_PATH), "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
