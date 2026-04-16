from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .config import (
    DEFAULT_INSTANCE_LABELS_JSON,
    DEFAULT_INSTANCE_MANIFEST_JSON,
    DEFAULT_LABEL_UI_PORT,
    DEFAULT_MANUAL_LABELS_CSV,
    DEFAULT_SUNRGBD_ROOT,
)
from .labeling_data import (
    build_instance_manifest,
    load_instance_manifest,
    load_or_create_label_store,
    save_label_store,
    write_instance_manifest,
)


def ensure_manifest(dataset_root: Path, seed_csv: Path, manifest_path: Path) -> None:
    if manifest_path.exists():
        return
    manifest = build_instance_manifest(dataset_root, seed_csv)
    write_instance_manifest(manifest, manifest_path)


def _load_web_index() -> bytes:
    web_path = Path(__file__).resolve().parent / "webapp" / "index.html"
    return web_path.read_bytes()


class LabelServer:
    def __init__(
        self,
        *,
        dataset_root: Path,
        seed_csv: Path,
        manifest_path: Path,
        label_store_path: Path,
        split_filter: str | None,
    ) -> None:
        ensure_manifest(dataset_root, seed_csv, manifest_path)
        manifest = load_instance_manifest(manifest_path)
        if split_filter is not None:
            manifest = [scene for scene in manifest if scene.get("split") == split_filter]

        self.dataset_root = dataset_root
        self.manifest = manifest
        self.label_store_path = label_store_path
        self.label_store = load_or_create_label_store(label_store_path, manifest)
        self.web_index = _load_web_index()
        self._lock = threading.Lock()

    def bootstrap_payload(self) -> dict:
        return {
            "manifest": self.manifest,
            "labels": self.label_store,
        }

    def scene_image_bytes(self, relpath: str) -> tuple[bytes, str] | None:
        safe_path = (self.dataset_root / relpath).resolve()
        if not safe_path.is_file():
            return None
        try:
            safe_path.relative_to(self.dataset_root.resolve())
        except ValueError:
            return None

        suffix = safe_path.suffix.lower()
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
        }.get(suffix, "application/octet-stream")
        return safe_path.read_bytes(), content_type

    def update_scene_annotation(self, scene_id: str, annotation: dict) -> None:
        with self._lock:
            self.label_store["scene_annotations"][scene_id] = annotation
            save_label_store(self.label_store_path, self.label_store)


def _build_handler(server_state: LabelServer):
    class LabelHTTPRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_bytes(server_state.web_index, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/bootstrap":
                payload = json.dumps(server_state.bootstrap_payload()).encode("utf-8")
                self._send_bytes(payload, "application/json; charset=utf-8")
                return
            if parsed.path == "/api/image":
                relpath = parse_qs(parsed.query).get("path", [""])[0]
                relpath = unquote(relpath)
                image_data = server_state.scene_image_bytes(relpath)
                if image_data is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                    return
                body, content_type = image_data
                self._send_bytes(body, content_type)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/save-scene":
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(content_length)
            data = json.loads(payload.decode("utf-8"))
            scene_id = str(data["scene_id"])
            annotation = data["annotation"]
            server_state.update_scene_annotation(scene_id, annotation)
            self._send_bytes(b'{"ok": true}', "application/json; charset=utf-8")

        def log_message(self, format: str, *args) -> None:
            return

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return LabelHTTPRequestHandler


def run_label_tool(
    *,
    dataset_root: Path = DEFAULT_SUNRGBD_ROOT,
    seed_csv: Path = DEFAULT_MANUAL_LABELS_CSV,
    manifest_path: Path = DEFAULT_INSTANCE_MANIFEST_JSON,
    label_store_path: Path = DEFAULT_INSTANCE_LABELS_JSON,
    split_filter: str | None = None,
    port: int = DEFAULT_LABEL_UI_PORT,
    open_browser: bool = False,
) -> None:
    state = LabelServer(
        dataset_root=dataset_root,
        seed_csv=seed_csv,
        manifest_path=manifest_path,
        label_store_path=label_store_path,
        split_filter=split_filter,
    )
    handler = _build_handler(state)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Label UI running at {url}")
    print("Press Ctrl+C in this terminal when you're done.")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down label UI...")
    finally:
        httpd.server_close()
