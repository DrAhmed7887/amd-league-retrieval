#!/usr/bin/env python3
"""Container CMD: load the model once, then serve app.py over a unix socket forever.

If loading fails, the process stays alive anyway: the harness execs into a running
container, and app.py turns an unreachable model into valid (empty) answers rather
than a crash.
"""

from __future__ import annotations

import sys
import time

from pathlib import Path

from engine.client import DEFAULT_SOCKET, failed_marker
from engine.server import serve


def main() -> None:
    started = time.monotonic()
    try:
        from engine.model import VisionLanguageModel

        model = VisionLanguageModel()
        print(f"model loaded in {time.monotonic() - started:.1f} s; serving on {DEFAULT_SOCKET}", file=sys.stderr, flush=True)
        serve(model, DEFAULT_SOCKET)
    except Exception as err:
        print(f"model service failed: {err.__class__.__name__}: {err}", file=sys.stderr, flush=True)
        Path(failed_marker(DEFAULT_SOCKET)).write_text(f"{err.__class__.__name__}: {err}\n")
    while True:  # keep the container up so the harness can still exec into it
        time.sleep(3600)


if __name__ == "__main__":
    main()
