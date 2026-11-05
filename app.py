#!/usr/bin/env python3
"""AMD AI League Match 3 entry point. The harness calls it in two shapes:

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

and reads /app/output/<query-id>_output.json after each question. That file is
always written with both graded keys, even if something fails, because a missing
key scores zero while an empty answer at least wins the unanswerable questions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from engine.answer import answer_question
from engine.index import build_index, load_index
from engine.readers import get_reader

OUTPUT_DIR = Path(os.environ.get("MC3_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
READER = os.environ.get("MC3_READER", "none")


def write_output(query_id: str, answer: str, citations, confidence: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"answer": answer, "citations": list(citations), "confidence": confidence}, ensure_ascii=False)
    tmp = OUTPUT_DIR / f".{query_id}_output.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(OUTPUT_DIR / f"{query_id}_output.json")  # whole file appears at once


def run_query(corpus: Path, query_id: str, query: str) -> None:
    answer, citations, confidence = "", [], 0.0
    try:
        try:
            index = load_index(INDEX_DIR)
        except (OSError, ValueError):
            print("index missing or unreadable; rebuilding in-process", file=sys.stderr)
            index = build_index(corpus, INDEX_DIR)
        answer, citations, confidence = answer_question(index, query, get_reader(READER))
    except Exception as err:
        print(f"query {query_id} failed: {err.__class__.__name__}: {err}", file=sys.stderr)
    finally:
        write_output(query_id, answer, citations, confidence)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index = build_index(args.index, INDEX_DIR)
        print(f"indexed {len(index['chunks'])} chunks; {len(index['images'])} images; "
              f"skipped {[(s['file'], s['reason']) for s in index['skipped']]}", file=sys.stderr)
        return 0
    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")
    run_query(args.corpus, args.query_id, args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
