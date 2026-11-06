#!/usr/bin/env python3
"""Match 4 live check: real questions against the live web, read by the resident model.

Needs network and a running model service (python3 server.py) for --reader qwen.
The expected values were checked by hand on 6 Nov 2026; live pages can change.

    python3 scripts/m4_check.py --reader qwen
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.answer import normalize  # noqa: E402
from engine.readers import get_reader  # noqa: E402
from engine.webanswer import answer_web  # noqa: E402

CASES = [
    ("Which year was the Python Software Foundation founded?", None, "2001"),
    ("What is the phone number for individual tax help?", "https://www.irs.gov/help/telephone-assistance", "800-829-1040"),
    ("What is the price of A Light in the Attic?", "https://books.toscrape.com", "£51.77"),
    ("Who is the CEO of AMD?", None, "Lisa Su"),
    ("What is the default port of PostgreSQL?", None, "5432"),
    ("In which city is the headquarters of RWTH Aachen University?", None, "Aachen"),
    ("What is the capital of Australia?", None, "Canberra"),
    ("Who wrote the novel Things Fall Apart?", None, "Chinua Achebe"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="qwen-web", help="qwen-web (model service) or none")
    args = ap.parse_args()
    reader = get_reader("qwen-web" if args.reader == "qwen" else args.reader)
    ok, worst = 0, 0.0
    for query, url, want in CASES:
        t = time.monotonic()
        value, cites, _ = answer_web(query, reader, url=url)
        secs = time.monotonic() - t
        worst = max(worst, secs)
        hit = normalize(value) == normalize(want) and bool(cites)
        ok += hit
        print(f"{'PASS' if hit else 'FAIL'} {secs:5.1f}s  {query[:50]:<50} -> {value!r:<18} {cites[:2]}", flush=True)
    print(f"\n{ok}/{len(CASES)} exact (grader-style normalisation) with a cited page; slowest {worst:.1f}s (limit 30 s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
