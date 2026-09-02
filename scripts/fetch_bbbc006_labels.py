"""Fetch the official BBBC006 reference-label archive with SHA-256 verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-assets.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "BBBC006"
    / "v1"
    / "archives"
    / "BBBC006_v1_labels.zip"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    asset = json.loads(MANIFEST_PATH.read_text())["reference_labels"]
    expected_sha256 = asset["sha256"]
    if OUTPUT_PATH.exists():
        if sha256_path(OUTPUT_PATH) != expected_sha256:
            raise ValueError(f"Existing label archive fails SHA-256: {OUTPUT_PATH}")
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial = OUTPUT_PATH.with_suffix(".zip.partial")
        if partial.exists():
            raise FileExistsError(
                f"Incomplete prior download exists: {partial}; inspect it before retrying"
            )
        request = Request(
            asset["url"], headers={"User-Agent": "TessScope/0.1 reproducibility"}
        )
        with urlopen(request, timeout=120) as response, partial.open("xb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
        if sha256_path(partial) != expected_sha256:
            raise ValueError(f"Downloaded label archive fails SHA-256: {partial}")
        partial.replace(OUTPUT_PATH)
    print(
        json.dumps(
            {
                "status": "verified",
                "sha256": expected_sha256,
                "output": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
