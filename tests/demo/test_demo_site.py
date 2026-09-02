"""Tests for the dependency-free local judge experience."""

from __future__ import annotations

import functools
import http.server
import json
import subprocess
import sys
import threading
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tessscope.demo.evidence import sha256_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = PROJECT_ROOT / "outputs" / "demo"
INDEX_PATH = DEMO_ROOT / "index.html"
DATA_PATH = DEMO_ROOT / "site-data.json"
MANIFEST_PATH = DEMO_ROOT / "site-manifest.json"


class AssetParser(HTMLParser):
    """Collect local document assets and basic semantic elements."""

    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.tags.append((tag, values))
        if tag in {"img", "script"} and values.get("src"):
            self.assets.append(str(values["src"]))
        if tag == "link" and values.get("href"):
            self.assets.append(str(values["href"]))


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    """Suppress expected localhost request logs during the smoke test."""

    def log_message(self, format: str, *args: object) -> None:
        return


def test_site_manifest_is_cached_local_and_hash_verified() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["status"] == "complete_local_static_judge_demo"
    assert manifest["mode"] == "cached_replay_not_live_inference"
    assert manifest["test_accessed"] is False
    assert manifest["entrypoint"] == "python scripts/serve_demo.py"
    assert manifest["network_binding"] == "127.0.0.1"
    assert manifest["deployment"] == "not_performed"
    for output in manifest["site_files"].values():
        path = PROJECT_ROOT / output["path"]
        assert output["sha256"] == sha256_path(path)
        assert output["size_bytes"] == path.stat().st_size
    for path_text, output in manifest["assets"].items():
        assert output["sha256"] == sha256_path(PROJECT_ROOT / path_text)


def test_site_data_preserves_frame_and_claim_traceability() -> None:
    data = json.loads(DATA_PATH.read_text())
    assert data["status"] == "cached_validation_replay_not_live"
    assert data["test_accessed"] is False
    assert data["selected_field"] == "n21_s1"
    assert data["selected_display_depth_um"] == -2.0
    assert len(data["frames"]) == 7
    assert sum(frame["representative_still"] for frame in data["frames"]) == 1
    assert {len(frame["systems"]) for frame in data["frames"]} == {3}
    for frame in data["frames"]:
        assert sha256_path(PROJECT_ROOT / "outputs" / "demo" / frame["path"]) == frame[
            "path_sha256"
        ]
        for system in frame["systems"]:
            source = PROJECT_ROOT / system["source"]
            assert sha256_path(source) == system["source_sha256"]
            assert system["before_pointer"].startswith("/rows/")
            if frame["depth_um"] == 0.0:
                assert system["after_pq"] is None
                assert system["corrected_pointer"] is None
            else:
                assert isinstance(system["after_pq"], float)
                assert system["corrected_pointer"].startswith("/corrected_rows/")

    headline = data["headline"]
    assert headline["exact_vs_stopped_pq"] == pytest.approx(0.006258840610359453)
    assert headline["ci_lower_95"] > 0.0
    assert headline["piecewise_ci_lower_95"] < 0.0 < headline["piecewise_ci_upper_95"]


def test_html_is_semantic_accessible_and_uses_only_local_assets() -> None:
    html = INDEX_PATH.read_text()
    parser = AssetParser()
    parser.feed(html)
    assert '<html lang="en">' in html
    assert '<main id="main-content">' in html
    assert "Cached replay · no computation occurs here" in html
    assert "PQ means panoptic quality" in html
    assert "The locked BBBC006 test" in html
    assert "Physical microscope performance" in html
    assert "did not significantly beat piecewise-028" in html
    depth_button_count = sum(
        tag == "button" and attrs.get("data-frame") is not None for tag, attrs in parser.tags
    )
    assert depth_button_count == 7
    for tag, attrs in parser.tags:
        if tag == "img":
            assert attrs.get("alt")
    for asset in parser.assets:
        assert "://" not in asset
        assert (DEMO_ROOT / asset).is_file()


def test_css_supports_small_screens_focus_and_reduced_motion() -> None:
    css = (DEMO_ROOT / "site.css").read_text()
    assert "@media (max-width: 560px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ":focus-visible" in css
    assert "min-height: 2.75rem" in css


def test_one_command_check_and_real_local_http_response() -> None:
    check = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "serve_demo.py"), "--check"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "status=complete_local_static_judge_demo" in check.stdout

    assert "test_accessed=false" in check.stdout

    handler = functools.partial(SilentHandler, directory=str(DEMO_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/", timeout=3
        ) as response:
            payload = response.read().decode()
            assert response.status == 200
            assert "Exact gradients improve closed-loop microscope correction" in payload
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/site-data.json", timeout=3
        ) as response:
            assert json.loads(response.read())["test_accessed"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
