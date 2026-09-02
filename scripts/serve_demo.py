"""Serve the cached TessScope judge demo with one dependency-free command."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "outputs" / "demo"
REQUIRED = (
    "index.html",
    "site.css",
    "app.js",
    "site-data.json",
    "site-manifest.json",
    "figure-manifest.json",
)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static evidence while keeping the terminal output concise."""

    def log_message(self, format: str, *args: object) -> None:
        return


def validate_demo() -> dict:
    """Validate the complete cached bundle without opening data or model files."""
    missing = [name for name in REQUIRED if not (DEMO_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Demo bundle is incomplete: {missing}")
    manifest = json.loads((DEMO_ROOT / "site-manifest.json").read_text())
    if manifest["status"] != "complete_local_static_judge_demo":
        raise ValueError("Demo manifest is not complete")
    if manifest["mode"] != "cached_replay_not_live_inference":
        raise ValueError("Demo must remain an explicit cached replay")
    if manifest["test_accessed"]:
        raise ValueError("Demo manifest reports locked-test access")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Local port; default 8765")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser")
    parser.add_argument("--check", action="store_true", help="Validate the bundle and exit")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    manifest = validate_demo()
    if args.check:
        print(f"status={manifest['status']}")
        print(f"mode={manifest['mode']}")
        print("test_accessed=false")
        return

    handler = functools.partial(QuietHandler, directory=str(DEMO_ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print("TessScope judge demo")
    print("Mode: cached validation replay; no live inference or test access")
    print(f"Open: {url}")
    print("Press Control-C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
