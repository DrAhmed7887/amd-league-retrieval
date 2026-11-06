#!/usr/bin/env python3
"""Score the Match 3 sample questions against a RUNNING container, the way the grader calls it.

Usage: python3 scripts/score_container.py <container-name>
The container must already have the sample corpus at /app/corpus and the model loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_sample import is_correct  # noqa: E402


def docker(*args, timeout=600):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def main() -> int:
    name = sys.argv[1]
    questions = json.loads((ROOT / "kit" / "sample-questions.json").read_text())["queries"]
    started = time.monotonic()
    index = docker("exec", name, "python3", "/app/app.py", "--index", "/app/corpus")
    print(f"index: rc={index.returncode} {time.monotonic() - started:.1f}s  {index.stderr.strip()[-300:]}")
    total = 0
    for q in questions:
        qid = f"query_{q['n']:02d}"
        t = time.monotonic()
        docker("exec", name, "python3", "/app/app.py", "--corpus", "/app/corpus", "--query-id", qid, "--query", q["query"], timeout=60)
        secs = time.monotonic() - t
        out = json.loads(docker("exec", name, "cat", f"/app/output/{qid}_output.json").stdout or "{}")
        ok_a, ok_c = is_correct(out.get("answer", ""), out.get("citations", []), q)
        score = 20 if ok_a and ok_c else 0
        total += score
        flag = "  OVER 30 s" if secs > 30 else ""
        print(f"{q['n']:>2} {score:>3}  {secs:5.1f}s  {out.get('answer', '')!r:<18} {out.get('citations')}{flag}")
    print(f"TOTAL {total}/200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
