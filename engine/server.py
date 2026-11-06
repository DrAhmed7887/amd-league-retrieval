"""Unix-socket service that keeps one model loaded for the whole run.

The harness starts a new process for every question, and loading the model takes
about 23 s, so loading per question would break the 30 s limit. The container's CMD
starts this service once; app.py talks to it.

Protocol: one JSON object per connection, newline-terminated, in both directions.
  request  {"op": "ping"} or {"op": "ask", "prompt": str, "image": str|null, "max_new_tokens": int}
  response {"ok": true, "text": str} or {"ok": false, "error": str}
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any, Dict

MAX_REQUEST_BYTES = 4 * 1024 * 1024


def _handle(model: Any, request: Dict) -> Dict:
    op = request.get("op")
    if op == "ping":
        return {"ok": True, "text": "pong"}
    if op != "ask":
        return {"ok": False, "error": f"unknown op {op!r}"}
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return {"ok": False, "error": "prompt must be a non-empty string"}
    image = request.get("image")
    max_new = int(request.get("max_new_tokens") or 64)
    try:
        return {"ok": True, "text": model.ask(prompt, image, max_new)}
    except Exception as err:  # one bad request must not kill the service
        return {"ok": False, "error": f"{err.__class__.__name__}: {err}"}


def _read_line(conn: socket.socket) -> bytes:
    data = b""
    while not data.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_REQUEST_BYTES:
            raise ValueError("request too large")
    return data


def serve(model: Any, socket_path: str) -> None:
    """Answer requests one at a time, forever."""
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(8)
    while True:
        conn, _ = listener.accept()
        with conn:
            try:
                request = json.loads(_read_line(conn).decode("utf-8"))
                response = _handle(model, request) if isinstance(request, dict) else {"ok": False, "error": "bad request"}
            except Exception as err:
                response = {"ok": False, "error": f"{err.__class__.__name__}: {err}"}
            try:
                conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            except OSError as err:
                print(f"client went away: {err}", file=sys.stderr)
