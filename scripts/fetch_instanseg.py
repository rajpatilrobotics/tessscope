"""Fetch and verify the frozen InstanSeg model bundle from its official release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "instanseg-v0.1.2.json"
ARCHIVE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "external"
    / "instanseg"
    / "downloads"
    / "single_channel_nuclei.zip"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts" / "external" / "instanseg" / "model-v0.1.2"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, expected_sha256: str) -> None:
    if ARCHIVE_PATH.exists():
        if sha256_path(ARCHIVE_PATH) != expected_sha256:
            raise ValueError(f"Existing archive fails SHA-256: {ARCHIVE_PATH}")
        return
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE_PATH.with_suffix(".zip.partial")
    if partial.exists():
        raise FileExistsError(
            f"Incomplete prior download exists: {partial}; inspect it before retrying"
        )
    request = Request(url, headers={"User-Agent": "TessScope/0.1 reproducibility"})
    with urlopen(request, timeout=120) as response, partial.open("xb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
    if sha256_path(partial) != expected_sha256:
        raise ValueError(f"Downloaded archive fails SHA-256: {partial}")
    partial.replace(ARCHIVE_PATH)


def safe_members(zip_file: ZipFile, expected_files: set[str]) -> dict[str, str]:
    matches: dict[str, str] = {}
    for name in zip_file.namelist():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe ZIP member: {name}")
        if path.name in expected_files:
            if path.name in matches:
                raise ValueError(f"Duplicate expected ZIP member: {path.name}")
            matches[path.name] = name
    missing = expected_files - matches.keys()
    if missing:
        raise ValueError(f"Model archive is missing expected files: {sorted(missing)}")
    return matches


def extract_verified(expected_files: dict[str, str]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE_PATH) as zip_file:
        members = safe_members(zip_file, set(expected_files))
        for filename, expected_sha256 in expected_files.items():
            output = OUTPUT_ROOT / filename
            if output.exists():
                if sha256_path(output) != expected_sha256:
                    raise ValueError(f"Existing model file fails SHA-256: {output}")
                continue
            payload = zip_file.read(members[filename])
            if hashlib.sha256(payload).hexdigest() != expected_sha256:
                raise ValueError(f"ZIP member fails SHA-256: {filename}")
            partial = output.with_suffix(output.suffix + ".partial")
            if partial.exists():
                raise FileExistsError(
                    f"Incomplete prior extraction exists: {partial}; inspect it before retrying"
                )
            partial.write_bytes(payload)
            partial.replace(output)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    asset = manifest["asset"]
    expected_files = manifest["files"]
    download_archive(asset["url"], asset["sha256"])
    extract_verified(expected_files)
    verified = {
        name: sha256_path(OUTPUT_ROOT / name) == expected_sha256
        for name, expected_sha256 in expected_files.items()
    }
    if not all(verified.values()):
        raise SystemExit("InstanSeg model verification failed")
    print(
        json.dumps(
            {
                "status": "verified",
                "release": asset["release"],
                "files": len(verified),
                "output": str(OUTPUT_ROOT.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
