"""Audit the local TessScope tree and Git history for public-release blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITHUB_FILE_LIMIT = 100 * 1024 * 1024
GITHUB_WARNING_SIZE = 50 * 1024 * 1024
MAX_SECRET_SCAN_SIZE = 20 * 1024 * 1024
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
ALLOWED_LARGE_EVIDENCE = Path("artifacts/runs/demo/validation-replay.npz")
FORBIDDEN_DATA_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".hdf5",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tif",
    ".tiff",
    ".zip",
}
SECRET_PATTERNS = {
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws-access-key": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github-token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "openai-api-key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "slack-token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "credential-assignment": re.compile(
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b"
        rb"\s*[:=]\s*['\"][^'\"\r\n]{12,}['\"]"
    ),
    "credential-in-url": re.compile(rb"https?://[^/\s:@]+:[^@\s/]+@"),
}
ABSOLUTE_PATH_PATTERN = re.compile(
    rb"(?:/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/|[A-Za-z]:\\Users\\)"
)
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
DATA_URI_PATTERN = re.compile(rb"data:[^;\s]+;base64,[A-Za-z0-9+/=\r\n]+")


@dataclass(frozen=True)
class Finding:
    """One redacted release-audit finding."""

    kind: str
    path: str
    detail: str


def _git(*arguments: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        input=input_text,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _tracked_paths() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [Path(value.decode()) for value in output.split(b"\0") if value]


def _history_blobs() -> list[tuple[int, str, str]]:
    objects = _git("rev-list", "--objects", "--all").splitlines()
    paths_by_object: dict[str, str] = {}
    object_ids = []
    for row in objects:
        object_id, _, path = row.partition(" ")
        object_ids.append(object_id)
        if path:
            paths_by_object.setdefault(object_id, path)
    batch_input = "".join(f"{object_id}\n" for object_id in object_ids)
    rows = _git(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_text=batch_input,
    ).splitlines()
    blobs = []
    for row in rows:
        object_id, object_type, size_text = row.split()
        if object_type == "blob":
            blobs.append((int(size_text), object_id, paths_by_object.get(object_id, "<unknown>")))
    return blobs


def _text_payload(data: bytes) -> bytes | None:
    if len(data) > MAX_SECRET_SCAN_SIZE or b"\0" in data[:8192]:
        return None
    return DATA_URI_PATTERN.sub(b"", data)


def _scan_payload(data: bytes, path: str, *, history: str = "") -> list[Finding]:
    payload = _text_payload(data)
    if payload is None:
        return []
    suffix = f" at Git blob {history}" if history else ""
    findings = [
        Finding(kind=name, path=path, detail=f"pattern match{suffix}; value redacted")
        for name, pattern in SECRET_PATTERNS.items()
        if pattern.search(payload)
    ]
    if ABSOLUTE_PATH_PATTERN.search(payload):
        findings.append(
            Finding(
                kind="absolute-private-path",
                path=path,
                detail=f"path match{suffix}; value redacted",
            )
        )
    return findings


def _scan_current(paths: list[Path]) -> tuple[list[Finding], list[Finding]]:
    content_findings = []
    file_findings = []
    for relative in paths:
        path = PROJECT_ROOT / relative
        if relative.suffix.lower() in FORBIDDEN_DATA_SUFFIXES:
            file_findings.append(
                Finding(
                    kind="forbidden-binary-or-archive",
                    path=str(relative),
                    detail="raw dataset, model, credential, or archive suffix",
                )
            )
        if path.is_file():
            content_findings.extend(_scan_payload(path.read_bytes(), str(relative)))
    return content_findings, file_findings


def _scan_history(blobs: list[tuple[int, str, str]]) -> list[Finding]:
    findings = []
    for size, object_id, path in blobs:
        if size > MAX_SECRET_SCAN_SIZE:
            continue
        payload = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        findings.extend(_scan_payload(payload, path, history=object_id[:12]))
    return findings


def _markdown_link_findings(paths: list[Path]) -> list[Finding]:
    findings = []
    for relative in paths:
        if relative.suffix.lower() not in {".md", ".markdown"}:
            continue
        text = (PROJECT_ROOT / relative).read_text(errors="replace")
        for match in MARKDOWN_LINK_PATTERN.finditer(text):
            raw_target = match.group(1).strip()
            if raw_target.startswith("<") and ">" in raw_target:
                target = raw_target[1 : raw_target.index(">")]
            else:
                target = raw_target.split(maxsplit=1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if target.startswith("/"):
                findings.append(
                    Finding("absolute-markdown-link", str(relative), "repository link is absolute")
                )
                continue
            destination = (PROJECT_ROOT / relative.parent / target).resolve()
            if not destination.exists():
                line = text.count("\n", 0, match.start()) + 1
                findings.append(
                    Finding("broken-markdown-link", str(relative), f"line {line}: {target}")
                )
    return findings


def audit(*, include_history_secrets: bool) -> dict:
    """Return a redacted, machine-readable release audit."""
    paths = _tracked_paths()
    blobs = _history_blobs()
    current_findings, forbidden_files = _scan_current(paths)
    history_findings = _scan_history(blobs) if include_history_secrets else []
    link_findings = _markdown_link_findings(paths)
    current_sizes = [
        ((PROJECT_ROOT / path).stat().st_size, str(path))
        for path in paths
        if (PROJECT_ROOT / path).is_file()
    ]
    current_oversized = [
        Finding("github-file-limit", path, f"{size} bytes")
        for size, path in current_sizes
        if size >= GITHUB_FILE_LIMIT
    ]
    history_oversized = [
        Finding("github-history-limit", path, f"{size} bytes at Git blob {object_id[:12]}")
        for size, object_id, path in blobs
        if size >= GITHUB_FILE_LIMIT
    ]
    warning_files = [
        {"path": path, "size_bytes": size}
        for size, path in sorted(current_sizes, reverse=True)
        if size >= GITHUB_WARNING_SIZE
    ]
    replay = PROJECT_ROOT / ALLOWED_LARGE_EVIDENCE
    license_path = PROJECT_ROOT / "LICENSE"
    trace_path = PROJECT_ROOT / "outputs" / "demo" / "traceability-manifest.json"
    required = [
        PROJECT_ROOT / "README.md",
        license_path,
        PROJECT_ROOT / "NOTICE",
        PROJECT_ROOT / "THIRD_PARTY_NOTICES.md",
        PROJECT_ROOT / "uv.lock",
    ]
    required_missing = [
        str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()
    ]
    license_hash = (
        hashlib.sha256(license_path.read_bytes()).hexdigest()
        if license_path.is_file()
        else None
    )
    trace = json.loads(trace_path.read_text()) if trace_path.is_file() else {}
    blockers = (
        current_findings
        + forbidden_files
        + history_findings
        + link_findings
        + current_oversized
        + history_oversized
    )
    if required_missing:
        blockers.append(
            Finding(
                "missing-required-file",
                ", ".join(required_missing),
                "release file absent",
            )
        )
    if license_hash != APACHE_2_SHA256:
        blockers.append(Finding("license-mismatch", "LICENSE", "not byte-identical to Apache-2.0"))
    if not replay.is_file():
        blockers.append(
            Finding(
                "missing-replay",
                str(ALLOWED_LARGE_EVIDENCE),
                "judge evidence absent",
            )
        )
    if trace.get("test_accessed") is not False:
        blockers.append(
            Finding(
                "demo-test-boundary",
                str(trace_path),
                "expected test_accessed=false",
            )
        )
    remotes = [line for line in _git("remote", "-v").splitlines() if line]
    return {
        "status": "GO" if not blockers else "NO-GO",
        "tracked_file_count": len(paths),
        "tracked_bytes": sum(size for size, _ in current_sizes),
        "git_blob_count": len(blobs),
        "largest_current_file": {
            "path": max(current_sizes)[1],
            "size_bytes": max(current_sizes)[0],
        },
        "largest_history_blob": {
            "path": max(blobs)[2],
            "size_bytes": max(blobs)[0],
        },
        "github_warning_files": warning_files,
        "replay": {
            "path": str(ALLOWED_LARGE_EVIDENCE),
            "size_bytes": replay.stat().st_size if replay.is_file() else None,
        },
        "history_secret_scan": include_history_secrets,
        "remote_configured": bool(remotes),
        "blockers": [asdict(finding) for finding in blockers],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-secrets",
        action="store_true",
        help="scan every text-sized Git blob for redacted secret patterns",
    )
    parser.add_argument("--json", action="store_true", help="print structured JSON")
    arguments = parser.parse_args()
    result = audit(include_history_secrets=arguments.history_secrets)
    if arguments.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"status={result['status']}")
        print(f"tracked_files={result['tracked_file_count']}")
        print(f"tracked_bytes={result['tracked_bytes']}")
        print(f"git_blobs={result['git_blob_count']}")
        print(f"history_secret_scan={str(result['history_secret_scan']).lower()}")
        print(f"remote_configured={str(result['remote_configured']).lower()}")
        print(f"blockers={len(result['blockers'])}")
        for finding in result["blockers"]:
            print(f"- {finding['kind']}: {finding['path']} ({finding['detail']})")
    raise SystemExit(0 if result["status"] == "GO" else 1)


if __name__ == "__main__":
    main()
