#!/usr/bin/env python3
"""AMD AI League entry point. One image answers two matches; each harness calls it its own way.

Match 2 (OCR), once per image:
    python3 /app/app.py --input-image /path/to/plate.png
    -> /app/output/plate_output.json   {"text": "...", "confidence": 0.9}

Match 3 (RAG), once to index, then once per question:
    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
    -> /app/output/query_01_output.json   {"answer": "...", "citations": [...], "confidence": 0.9}

The model stays loaded in server.py (started by the container's CMD); this script is a
thin client. Output files are always written with every graded key, even on failure,
because a missing key scores zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from engine.answer import answer_question
from engine.client import ModelClient, ModelError
from engine.index import build_index, load_index
from engine.ocr import OCR_PROMPT, clean_ocr_text
from engine.readers import TRANSCRIBE_PROMPT, TRANSCRIBE_TOKENS, WebQwenReader, get_reader
from engine.webanswer import TOTAL_BUDGET as WEB_TOTAL_BUDGET, answer_web

OUTPUT_DIR = Path(os.environ.get("MC3_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
READER = os.environ.get("MC3_READER", "none")
# The index pass may start while the service is still loading the model (~25 s on the
# W7900D), so it waits longer than a question does. Both stay inside the harness budgets.
INDEX_WAIT_SECONDS = float(os.environ.get("MODEL_INDEX_WAIT_SECONDS", "300"))
OCR_TOKENS = 128  # 64 cut off a long label mid-sentence on the AMD notebook (6 Nov)


def write_json(stem: str, payload: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_DIR / f".{stem}_output.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUTPUT_DIR / f"{stem}_output.json")  # the whole file appears at once


def make_image_reader():
    """Transcribe images through the service; after one 'unreachable', stop waiting on it."""
    if READER != "qwen":
        return None
    client = ModelClient(wait_seconds=INDEX_WAIT_SECONDS)
    state = {"down": False}

    def read(path: str) -> str:
        if state["down"]:
            raise ModelError("model service is down")
        try:
            return client.ask(TRANSCRIBE_PROMPT, path, TRANSCRIBE_TOKENS)
        except ModelError as err:
            if "not reachable" in str(err):
                state["down"] = True
            raise

    return read


def run_index(corpus: Path) -> None:
    index = build_index(corpus, INDEX_DIR, image_reader=make_image_reader())
    print(f"indexed {len(index['chunks'])} chunks; {len(index['images'])} images; "
          f"skipped {[(s['file'], s['reason']) for s in index['skipped']]}", file=sys.stderr)


def run_query(corpus: Path, query_id: str, query: str) -> None:
    answer, citations, confidence = "", [], 0.0
    try:
        try:
            index = load_index(INDEX_DIR)
        except (OSError, ValueError):
            print("index missing or unreadable; rebuilding without images", file=sys.stderr)
            index = build_index(corpus, INDEX_DIR)
        answer, citations, confidence = answer_question(index, query, get_reader(READER))
    except Exception as err:
        print(f"query {query_id} failed: {err.__class__.__name__}: {err}", file=sys.stderr)
    finally:
        write_json(query_id, {"answer": answer, "citations": list(citations), "confidence": confidence})


def run_ocr(image: Path) -> None:
    text = ""
    try:
        text = clean_ocr_text(ModelClient().ask(OCR_PROMPT, str(image), OCR_TOKENS))
    except Exception as err:
        print(f"ocr {image.name} failed: {err.__class__.__name__}: {err}", file=sys.stderr)
    finally:
        write_json(image.stem, {"text": text, "confidence": 0.9 if text else 0.0})


def run_web(query_id: str, query: str, url: Optional[str], pages: Optional[Path]) -> None:
    """Match 4: answer from the web (or a folder of saved pages), citing page URLs."""
    start = time.monotonic()
    write_json(query_id, {"answer": "", "citations": [], "confidence": 0.0})  # a timeout still leaves a valid file
    answer, citations, confidence = "", [], 0.0
    try:
        if READER == "qwen":  # the model call gets whatever is left of the question's budget, never more
            reader = WebQwenReader(ModelClient(deadline=start + WEB_TOTAL_BUDGET))
        else:
            reader = get_reader(READER)
        answer, citations, confidence = answer_web(query, reader, url=url, corpus=pages, start=start)
    except Exception as err:
        print(f"web query {query_id} failed: {err.__class__.__name__}: {err}", file=sys.stderr)
    finally:
        payload = {"answer": answer, "citations": list(citations), "confidence": confidence}
        if citations:
            payload["source_url"] = citations[0]
        write_json(query_id, payload)


def _from_question_file(args) -> None:
    """--question-file FILE: a JSON object with the question fields, for harnesses that pass one."""
    data = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    args.query_id = args.query_id or data.get("query_id") or data.get("id") or data.get("qid")
    args.query = args.query or data.get("query") or data.get("question")
    args.url = args.url or data.get("url") or data.get("website") or data.get("start_url")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", type=Path, help="Match 2: read the text in this image")
    ap.add_argument("--index", type=Path, help="Match 3: build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="Match 3: corpus root for a query")
    ap.add_argument("--query-id", "--id", "--qid", dest="query_id", help="output stem the harness assigns")
    ap.add_argument("--query", "--question", dest="query", help="the question to answer")
    ap.add_argument("--url", "--website", "--start-url", dest="url", help="Match 4: optional start page")
    ap.add_argument("--pages", type=Path, help="Match 4: a folder of saved pages instead of the live web")
    # Not "--input": argparse would then stop expanding the "--input" prefix to --input-image.
    ap.add_argument("--question-file", help="Match 4: JSON file with query_id/query/url")
    args = ap.parse_args()

    if args.input_image is not None:
        run_ocr(args.input_image)
        return 0
    if args.index is not None:
        run_index(args.index)
        return 0
    if args.question_file:
        try:
            _from_question_file(args)
        except Exception as err:
            print(f"unreadable --question-file: {err}", file=sys.stderr)
    if args.query is None or not args.query_id:
        if args.query_id:  # still leave a valid, empty answer for the grader
            write_json(args.query_id, {"answer": "", "citations": [], "confidence": 0.0})
            return 0
        ap.error("a query needs --query-id and --query")
    if args.corpus is not None and args.url is None and os.environ.get("MATCH_MODE") != "web":
        run_query(args.corpus, args.query_id, args.query)  # Match 3, unchanged
        return 0
    run_web(args.query_id, args.query, args.url, args.pages)  # Match 4
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # the answer is written; don't let slow fetch threads hold the process past 30 s


if __name__ == "__main__":
    raise SystemExit(main())
