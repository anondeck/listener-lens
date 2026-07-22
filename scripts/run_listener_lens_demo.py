#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from earshift_bakeoff.listener_lens import (
    ListenerLensError,
    ListenerLensService,
    local_prerequisites,
)


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "prototype" / "listener-lens" / "index.html"
MAX_REQUEST_BYTES = 4096


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def make_handler(service: ListenerLensService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ListenerLensPrototype/0.1"

        def log_message(self, format: str, *args: object) -> None:
            sys.stderr.write("listener-lens: " + format % args + "\n")

        def _send_json(self, status: int, value: Any) -> None:
            body = _json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                body = INDEX_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/status":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        **service.status(),
                        "local_prerequisites": local_prerequisites(),
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path == "/api/render":
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {
                        "error": "renderer_pending_human_review",
                        "message": (
                            "The production renderer is intentionally disabled until "
                            "matched-pair and acoustic-realization gates are complete."
                        ),
                        "api_calls_made": 0,
                    },
                )
                return
            if self.path != "/api/transform":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if not 0 < content_length <= MAX_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "invalid_request_size"},
                )
                return
            try:
                body = json.loads(self.rfile.read(content_length))
                text = body["text"]
                profile_id = body.get("profile_id", "en-to-pt-BR-vowel-lens")
                if not isinstance(text, str) or not isinstance(profile_id, str):
                    raise ListenerLensError("Text and profile_id must be strings.")
                payload = service.transform(text, profile_id)
            except (json.JSONDecodeError, KeyError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_json", "message": "Send JSON with a text field."},
                )
                return
            except ListenerLensError as exc:
                self._send_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {"error": "unsupported_input", "message": str(exc)},
                )
                return
            self._send_json(HTTPStatus.OK, payload)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the zero-API typed listener-lens prototype."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8788, type=int)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    if not INDEX_PATH.is_file():
        raise SystemExit(f"Missing prototype page: {INDEX_PATH}")
    prerequisites = local_prerequisites()
    if not prerequisites["espeak_ng"]:
        raise SystemExit("espeak-ng is required for the local typed prototype")

    service = ListenerLensService()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    url = f"http://{args.host}:{args.port}/"
    print(f"Listener-lens prototype: {url}")
    print("No OpenAI API calls are enabled in this local prototype server.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
