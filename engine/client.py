"""Client for the resident model service (see engine/server.py)."""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Optional

DEFAULT_SOCKET = os.environ.get("MODEL_SOCKET", "/tmp/amd-league-model.sock")
RESPONSE_TIMEOUT = 120.0


class ModelError(RuntimeError):
    pass


def failed_marker(socket_path: str) -> str:
    """File the service writes when the model cannot load, so clients stop waiting at once."""
    return socket_path + ".failed"


class ModelClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET, wait_seconds: Optional[float] = None):
        self.socket_path = socket_path
        self.wait_seconds = float(os.environ.get("MODEL_WAIT_SECONDS", "20")) if wait_seconds is None else wait_seconds

    def _connect(self) -> socket.socket:
        """Retry until the service is listening: it may still be loading the model."""
        deadline = time.monotonic() + self.wait_seconds
        while True:
            if os.path.exists(failed_marker(self.socket_path)):
                raise ModelError("model service failed to load; see the container log")
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                conn.connect(self.socket_path)
                return conn
            except OSError as err:
                conn.close()
                if time.monotonic() >= deadline:
                    raise ModelError(f"model service not reachable at {self.socket_path}: {err}") from None
                time.sleep(0.25)

    def _call(self, request: dict) -> str:
        conn = self._connect()
        with conn:
            conn.settimeout(RESPONSE_TIMEOUT)
            conn.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
        try:
            response = json.loads(data.decode("utf-8"))
        except ValueError:
            raise ModelError("model service sent an unreadable reply") from None
        if not response.get("ok"):
            raise ModelError(response.get("error", "unknown model error"))
        return str(response.get("text", ""))

    def ping(self) -> None:
        self._call({"op": "ping"})

    def ask(self, prompt: str, image: Optional[str] = None, max_new_tokens: int = 64) -> str:
        return self._call({"op": "ask", "prompt": prompt, "image": image, "max_new_tokens": max_new_tokens})
