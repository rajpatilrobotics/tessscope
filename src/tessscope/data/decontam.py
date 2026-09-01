"""Identity-only BBBC039 versus BBBC038 decontamination.

This module deliberately does not load masks, predictions, or quality metrics.
It produces evidence for exact and near-duplicate identity review before any
held-out BBBC039 evaluation is run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from skimage.transform import resize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BBBC039_ROOT = PROJECT_ROOT / "data" / "external" / "BBBC039"
BBBC038_ROOT = PROJECT_ROOT / "data" / "external" / "BBBC038" / "stage1_train"
MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests"


@dataclass(frozen=True)
class ImageEvidence:
    """Hashes and identifiers used only for identity detection."""

    dataset: str
    image_id: str
    split: str | None
    relative_path: str
    file_sha256: str
    decoded_gray_sha256: str
    shape: tuple[int, int]
    dtype: str
    phash_d4: tuple[str, ...]


def file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_gray(path: Path) -> np.ndarray:
    """Decode an image and reduce channels to deterministic grayscale float32."""
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim == 2:
        gray = value
    elif value.ndim == 3:
        channels = value[..., :3]
        if channels.shape[-1] == 1 or np.all(channels == channels[..., :1]):
            gray = channels[..., 0]
        else:
            gray = (
                0.2126 * channels[..., 0]
                + 0.7152 * channels[..., 1]
                + 0.0722 * channels[..., 2]
            )
    else:
        raise ValueError(f"Unsupported image shape {value.shape} at {path}")
    return np.ascontiguousarray(gray, dtype=np.float32)


def decoded_gray_sha256(gray: np.ndarray) -> str:
    """Hash decoded canonical pixels with shape metadata."""
    value = np.ascontiguousarray(gray, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def normalized_view(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """Resize, center, and L2-normalize for illumination-invariant NCC."""
    value = resize(
        np.asarray(gray, dtype=np.float32),
        (size, size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    value -= value.mean(dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.zeros_like(value)
    return value / norm


def d4_views(value: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the eight rotations/reflections in deterministic order."""
    rotations = tuple(np.rot90(value, k) for k in range(4))
    reflected = np.fliplr(value)
    return rotations + tuple(np.rot90(reflected, k) for k in range(4))


def _phash(value: np.ndarray, low_frequency_size: int = 8) -> str:
    coefficients = dctn(value, type=2, norm="ortho")[:low_frequency_size, :low_frequency_size]
    payload = coefficients.reshape(-1)[1:]
    bits = payload > np.median(payload)
    packed = np.packbits(bits.astype(np.uint8), bitorder="big")
    return packed.tobytes().hex()


def perceptual_hashes(gray: np.ndarray) -> tuple[str, ...]:
    """Compute a pHash for every D4 orientation of a normalized 64×64 view."""
    return tuple(_phash(view) for view in d4_views(normalized_view(gray)))


def hamming_hex(left: str, right: str) -> int:
    """Return bitwise Hamming distance for equal-length hexadecimal hashes."""
    if len(left) != len(right):
        raise ValueError("Perceptual hashes must have equal lengths")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _read_split_names(metadata_dir: Path) -> dict[str, str]:
    split_by_stem: dict[str, str] = {}
    expected_counts = {"training": 100, "validation": 50, "test": 50}
    for split, expected in expected_counts.items():
        names = [line.strip() for line in (metadata_dir / f"{split}.txt").read_text().splitlines()]
        names = [name for name in names if name]
        if len(names) != expected:
            raise ValueError(f"Expected {expected} {split} names, found {len(names)}")
        for name in names:
            stem = Path(name).stem
            if stem in split_by_stem:
                raise ValueError(f"Duplicate BBBC039 split entry: {stem}")
            split_by_stem[stem] = split
    return split_by_stem


def discover_bbbc039(root: Path = BBBC039_ROOT) -> list[tuple[Path, str]]:
    """Return all 200 official images with their frozen split."""
    split_by_stem = _read_split_names(root / "metadata" / "metadata")
    paths = sorted(
        path
        for path in (root / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png"}
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )
    by_stem = {path.stem: path for path in paths}
    missing = sorted(set(split_by_stem) - set(by_stem))
    unexpected = sorted(set(by_stem) - set(split_by_stem))
    if missing or unexpected:
        raise ValueError(
            f"BBBC039 image/split mismatch: missing={missing}, unexpected={unexpected}"
        )
    return [(by_stem[stem], split_by_stem[stem]) for stem in sorted(split_by_stem)]


def discover_bbbc038(root: Path = BBBC038_ROOT) -> list[Path]:
    """Return the 670 stage-1 training images disclosed to train the observer."""
    paths = sorted(
        path
        for path in root.glob("*/images/*")
        if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff"}
    )
    if len(paths) != 670:
        raise ValueError(f"Expected 670 BBBC038 training images, found {len(paths)}")
    return paths


def _evidence(path: Path, dataset: str, split: str | None, root: Path) -> ImageEvidence:
    gray = canonical_gray(path)
    return ImageEvidence(
        dataset=dataset,
        image_id=path.stem if dataset == "BBBC039" else path.parent.parent.name,
        split=split,
        relative_path=str(path.relative_to(root)),
        file_sha256=file_sha256(path),
        decoded_gray_sha256=decoded_gray_sha256(gray),
        shape=(int(gray.shape[0]), int(gray.shape[1])),
        dtype=str(gray.dtype),
        phash_d4=perceptual_hashes(gray),
    )


def _minimum_phash_distance(left: ImageEvidence, right: ImageEvidence) -> int:
    return min(hamming_hex(a, b) for a in left.phash_d4 for b in right.phash_d4)


def build_decontamination(
    ncc_threshold: float = 0.995,
    phash_hamming_threshold: int = 4,
) -> dict[str, object]:
    """Build split/evidence manifests and flag identity candidates."""
    bbbc039_paths = discover_bbbc039()
    bbbc038_paths = discover_bbbc038()
    evidence_039 = [
        _evidence(path, "BBBC039", split, BBBC039_ROOT) for path, split in bbbc039_paths
    ]
    evidence_038 = [
        _evidence(path, "BBBC038", None, BBBC038_ROOT) for path in bbbc038_paths
    ]

    views_039 = np.stack([normalized_view(canonical_gray(path)) for path, _ in bbbc039_paths])
    views_038 = np.stack([normalized_view(canonical_gray(path)) for path in bbbc038_paths])
    matrix_038 = views_038.reshape(len(views_038), -1)
    best_ncc = np.full((len(views_039), len(views_038)), -1.0, dtype=np.float32)
    best_orientation = np.zeros(best_ncc.shape, dtype=np.uint8)
    for orientation in range(8):
        oriented = np.stack([d4_views(view)[orientation] for view in views_039])
        scores = oriented.reshape(len(oriented), -1) @ matrix_038.T
        update = scores > best_ncc
        best_ncc[update] = scores[update]
        best_orientation[update] = orientation

    candidates: list[dict[str, object]] = []
    for i, left in enumerate(evidence_039):
        for j, right in enumerate(evidence_038):
            exact_file = left.file_sha256 == right.file_sha256
            exact_decoded = left.decoded_gray_sha256 == right.decoded_gray_sha256
            ncc = float(best_ncc[i, j])
            phash_distance = _minimum_phash_distance(left, right)
            reasons = []
            if exact_file:
                reasons.append("file_sha256")
            if exact_decoded:
                reasons.append("decoded_gray_sha256")
            if ncc >= ncc_threshold:
                reasons.append("ncc")
            if phash_distance <= phash_hamming_threshold:
                reasons.append("phash")
            if reasons:
                candidates.append(
                    {
                        "bbbc039_id": left.image_id,
                        "bbbc039_split": left.split,
                        "bbbc039_path": left.relative_path,
                        "bbbc038_id": right.image_id,
                        "bbbc038_path": right.relative_path,
                        "reasons": reasons,
                        "file_sha256_equal": exact_file,
                        "decoded_gray_sha256_equal": exact_decoded,
                        "maximum_d4_ncc": ncc,
                        "bbbc039_d4_orientation": int(best_orientation[i, j]),
                        "minimum_phash_hamming": phash_distance,
                        "manual_identity_review": "pending",
                    }
                )

    split_counts = {
        split: sum(item.split == split for item in evidence_039)
        for split in ("training", "validation", "test")
    }
    return {
        "schema_version": 1,
        "purpose": "identity-only decontamination; no masks, predictions, or quality review",
        "thresholds": {
            "maximum_d4_ncc": ncc_threshold,
            "minimum_phash_hamming": phash_hamming_threshold,
        },
        "counts": {
            "BBBC039": len(evidence_039),
            "BBBC038_stage1_train": len(evidence_038),
            "BBBC039_splits": split_counts,
            "flagged_pairs": len(candidates),
        },
        "candidates": sorted(
            candidates,
            key=lambda item: (-float(item["maximum_d4_ncc"]), int(item["minimum_phash_hamming"])),
        ),
        "images": {
            "BBBC039": [asdict(item) for item in evidence_039],
            "BBBC038": [asdict(item) for item in evidence_038],
        },
    }


def write_manifests(report: dict[str, object], output_dir: Path = MANIFEST_ROOT) -> None:
    """Write reproducible JSON plus a compact candidate-review CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bbbc-decontamination.json").write_text(json.dumps(report, indent=2) + "\n")
    candidates = report["candidates"]
    fieldnames = [
        "bbbc039_id",
        "bbbc039_split",
        "bbbc039_path",
        "bbbc038_id",
        "bbbc038_path",
        "reasons",
        "file_sha256_equal",
        "decoded_gray_sha256_equal",
        "maximum_d4_ncc",
        "bbbc039_d4_orientation",
        "minimum_phash_hamming",
        "manual_identity_review",
    ]
    with (output_dir / "bbbc-decontamination-candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:  # type: ignore[union-attr]
            row = dict(candidate)
            row["reasons"] = ";".join(row["reasons"])
            writer.writerow(row)


def main() -> None:
    report = build_decontamination()
    write_manifests(report)
    print(json.dumps(report["counts"], indent=2))
    print(f"Review {len(report['candidates'])} flagged identity pairs before freezing exclusions.")


if __name__ == "__main__":
    main()
