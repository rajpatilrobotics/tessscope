"""Range-extract only BBBC006 Hoechst planes and unpack frozen reference assets."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZipInfo

from PIL import Image

from tessscope.v2.data.bbbc006 import (
    DATA_ROOT,
    IMAGE_ROOT,
    LABEL_ROOT,
    SPLIT_MANIFEST,
    Z_PLANES,
    all_wells,
    parse_image_filename,
    parse_label_filename,
)
from tessscope.v2.data.remote_zip import (
    HTTPRangeFile,
    atomic_write,
    fetch_member,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-assets.json"
MEMBER_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "v2" / "bbbc006-w1-members.json"
ARCHIVE_ROOT = DATA_ROOT / "archives"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planes", type=int, nargs="*", default=list(Z_PLANES))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("training", "validation", "test"),
        default=None,
        help=(
            "well splits to materialize; pass training validation for the public "
            "reproduction path without touching the sealed test split"
        ),
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=None,
        help="smoke-only cap per plane; capped runs do not write the final member manifest",
    )
    return parser.parse_args()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def member_index(
    plane: int, metadata: dict[str, Any]
) -> tuple[str, list[tuple[ZipInfo, str, int, Path]]]:
    archive = metadata["selected_planes"][str(plane)]
    url = archive["url"]
    remote = HTTPRangeFile(url, int(archive["content_length"]))
    with ZipFile(io.BufferedReader(remote)) as zip_file:
        selected = []
        seen: set[tuple[str, int]] = set()
        for info in zip_file.infolist():
            if info.is_dir():
                continue
            parsed = parse_image_filename(info.filename)
            if parsed.channel != 1:
                continue
            key = (parsed.well, parsed.site)
            if key in seen:
                raise ValueError(f"Duplicate Hoechst field in z{plane}: {key}")
            seen.add(key)
            output = IMAGE_ROOT / f"z_{plane:02d}" / f"{parsed.well}_s{parsed.site}_w1.tif"
            selected.append((info, parsed.well, parsed.site, output))
    expected = {(well, site) for well in all_wells() for site in (1, 2)}
    if seen != expected:
        raise ValueError(f"z{plane} has {len(seen)} Hoechst fields, expected 768")
    return url, sorted(selected, key=lambda value: (value[1], value[2]))


def existing_result(
    info: ZipInfo, plane: int, well: str, site: int, output: Path
) -> dict[str, Any] | None:
    if not output.exists():
        return None
    value = output.read_bytes()
    crc = binascii.crc32(value) & 0xFFFFFFFF
    if len(value) != info.file_size or crc != info.CRC:
        raise ValueError(f"Existing BBBC006 file fails ZIP integrity: {output}")
    return result_row(info, plane, well, site, output, value, status="existing")


def result_row(
    info: ZipInfo,
    plane: int,
    well: str,
    site: int,
    output: Path,
    value: bytes,
    *,
    status: str,
) -> dict[str, Any]:
    with Image.open(io.BytesIO(value)) as image:
        shape_yx = [image.height, image.width]
        mode = image.mode
    if shape_yx != [520, 696] or mode not in {"I;16", "I;16B", "I;16L"}:
        raise ValueError(f"Unexpected decoded BBBC006 TIFF: {shape_yx}, {mode}, {info.filename}")
    return {
        "z_plane": plane,
        "depth_um": float(2 * (plane - 16)),
        "well": well,
        "site": site,
        "source_member": info.filename,
        "relative_path": str(output.relative_to(PROJECT_ROOT)),
        "bytes": len(value),
        "zip_crc32": f"{info.CRC:08x}",
        "sha256": sha256(value),
        "status": status,
    }


def fetch_one(
    url: str,
    info: ZipInfo,
    plane: int,
    well: str,
    site: int,
    output: Path,
) -> dict[str, Any]:
    existing = existing_result(info, plane, well, site, output)
    if existing is not None:
        return existing
    value = fetch_member(url, info)
    row = result_row(info, plane, well, site, output, value, status="downloaded")
    atomic_write(output, value)
    return row


def fetch_plane(
    plane: int,
    metadata: dict[str, Any],
    *,
    workers: int,
    maximum_members: int | None,
    target_wells: set[str],
) -> list[dict[str, Any]]:
    url, members = member_index(plane, metadata)
    members = [member for member in members if member[1] in target_wells]
    if maximum_members is not None:
        members = members[:maximum_members]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_one, url, info, plane, well, site, output): (well, site)
            for info, well, site, output in members
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {"z_plane": plane, "completed": completed, "total": len(futures)}
                    ),
                    flush=True,
                )
    return sorted(rows, key=lambda row: (row["well"], row["site"]))


def extract_labels(target_wells: set[str]) -> list[dict[str, Any]]:
    archive = ARCHIVE_ROOT / "BBBC006_v1_labels.zip"
    rows = []
    seen: set[tuple[str, int]] = set()
    with ZipFile(archive) as zip_file:
        for info in zip_file.infolist():
            if info.is_dir():
                continue
            well, site = parse_label_filename(info.filename)
            if well not in target_wells:
                continue
            key = (well, site)
            if key in seen:
                raise ValueError(f"Duplicate BBBC006 reference labels: {key}")
            seen.add(key)
            value = zip_file.read(info)
            output = LABEL_ROOT / f"{well}_s{site}.png"
            if output.exists() and output.read_bytes() != value:
                raise ValueError(f"Existing label differs from official ZIP member: {output}")
            if not output.exists():
                atomic_write(output, value)
            rows.append(
                {
                    "well": well,
                    "site": site,
                    "source_member": info.filename,
                    "relative_path": str(output.relative_to(PROJECT_ROOT)),
                    "bytes": len(value),
                    "zip_crc32": f"{info.CRC:08x}",
                    "sha256": sha256(value),
                }
            )
    expected = {(well, site) for well in target_wells for site in (1, 2)}
    if seen != expected:
        raise ValueError(f"Reference-label archive has {len(seen)} fields, expected 768")
    return sorted(rows, key=lambda row: (row["well"], row["site"]))


def assert_matches_frozen_manifest(payload: dict[str, Any]) -> None:
    """Confirm a complete materialization against the committed member inventory."""
    frozen = json.loads(MEMBER_MANIFEST.read_text())
    image_keys = (
        "z_plane",
        "depth_um",
        "well",
        "site",
        "source_member",
        "relative_path",
        "bytes",
        "zip_crc32",
        "sha256",
    )
    label_keys = (
        "well",
        "site",
        "source_member",
        "relative_path",
        "bytes",
        "zip_crc32",
        "sha256",
    )

    def signatures(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[tuple]:
        return sorted(tuple(row[key] for key in keys) for row in rows)

    if signatures(payload["images"], image_keys) != signatures(
        frozen["images"], image_keys
    ):
        raise ValueError("Materialized BBBC006 images differ from the frozen manifest")
    if signatures(payload["reference_labels"], label_keys) != signatures(
        frozen["reference_labels"], label_keys
    ):
        raise ValueError("Materialized BBBC006 labels differ from the frozen manifest")


def main() -> None:
    args = parse_args()
    if not SPLIT_MANIFEST.exists():
        raise SystemExit("Freeze the BBBC006 well split before downloading field assets")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not args.planes or any(plane not in Z_PLANES for plane in args.planes):
        raise SystemExit(f"--planes must be a non-empty subset of {Z_PLANES}")
    split_payload = json.loads(SPLIT_MANIFEST.read_text())
    requested_splits = args.splits or ["training", "validation", "test"]
    target_wells = {
        well for split in requested_splits for well in split_payload["wells"][split]
    }
    metadata = json.loads(ASSET_MANIFEST.read_text())
    image_rows = []
    for plane in args.planes:
        image_rows.extend(
            fetch_plane(
                plane,
                metadata,
                workers=args.workers,
                maximum_members=args.max_members,
                target_wells=target_wells,
            )
        )
    label_rows = extract_labels(target_wells)
    complete = (
        args.max_members is None
        and set(args.planes) == set(Z_PLANES)
        and set(requested_splits) == {"training", "validation", "test"}
    )
    payload = {
        "dataset": "BBBC006v1",
        "status": "complete" if complete else "permitted_split_materialization",
        "splits": requested_splits,
        "images": image_rows,
        "reference_labels": label_rows,
        "image_count": len(image_rows),
        "label_count": len(label_rows),
    }
    if complete:
        if len(image_rows) != 7 * 768 or len(label_rows) != 768:
            raise ValueError("Complete BBBC006 fetch has the wrong asset count")
        if MEMBER_MANIFEST.exists():
            assert_matches_frozen_manifest(payload)
            payload["status"] = "verified_against_frozen_manifest"
        else:
            MEMBER_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        progress = (
            PROJECT_ROOT
            / "artifacts"
            / "runtime-runs"
            / "reproduction"
            / "fetch-progress.json"
        )
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(json.dumps(payload, indent=2) + "\n")
    summary = {
        key: payload[key] for key in ("status", "image_count", "label_count")
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
