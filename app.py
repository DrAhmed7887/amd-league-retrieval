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
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
_ID_RE = re.compile(r"^(?:find|query|task|q)[_-]?\d+$", re.IGNORECASE)


def write_json(
    stem: str,
    payload: dict,
    output_dir: Path = OUTPUT_DIR,
    output_file: Optional[Path] = None,
) -> None:
    if output_file is not None and (
        (output_file.exists() and output_file.is_dir()) or str(output_file).endswith(os.sep)
    ):
        target = output_file / f"{stem}_output.json"
    elif output_file is not None:
        target = output_file
    else:
        target = output_dir / f"{stem}_output.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)  # the whole file appears at once


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


def run_query(
    corpus: Path,
    query_id: str,
    query: str,
    output_dir: Path = OUTPUT_DIR,
    output_file: Optional[Path] = None,
) -> None:
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
        write_json(
            query_id,
            {"answer": answer, "citations": list(citations), "confidence": confidence},
            output_dir=output_dir,
            output_file=output_file,
        )


def run_ocr(
    image: Path,
    output_dir: Path = OUTPUT_DIR,
    output_file: Optional[Path] = None,
) -> None:
    text = ""
    try:
        text = clean_ocr_text(ModelClient().ask(OCR_PROMPT, str(image), OCR_TOKENS))
    except Exception as err:
        print(f"ocr {image.name} failed: {err.__class__.__name__}: {err}", file=sys.stderr)
    finally:
        write_json(
            image.stem,
            {"text": text, "confidence": 0.9 if text else 0.0},
            output_dir=output_dir,
            output_file=output_file,
        )


def run_web(
    query_id: str,
    query: str,
    url: Optional[str],
    pages: Optional[Path],
    output_dir: Path = OUTPUT_DIR,
    output_file: Optional[Path] = None,
) -> None:
    """Match 4: answer from the web (or a folder of saved pages), citing page URLs."""
    start = time.monotonic()
    empty = {"answer": "", "citations": [], "confidence": 0.0}
    write_json(query_id, empty, output_dir=output_dir, output_file=output_file)  # timeout still leaves valid file
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
        write_json(query_id, payload, output_dir=output_dir, output_file=output_file)


def _item_from_dict(data: Dict, default_id: Optional[str] = None) -> Dict[str, Optional[str]]:
    pages_val = data.get("pages")
    return {
        "query_id": data.get("query_id") or data.get("id") or data.get("qid")
        or data.get("find_id") or data.get("task_id") or data.get("name") or default_id,
        "query": data.get("query") or data.get("question") or data.get("task") or data.get("prompt"),
        "url": data.get("url") or data.get("website") or data.get("start_url"),
        "pages": str(pages_val) if pages_val else None,
    }


def _load_file_items(path: Path) -> List[Dict[str, Optional[str]]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except ValueError:
        items = []
        for idx, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(_item_from_dict(obj, default_id=f"{path.stem}_{idx:02d}"))
        return items
    if isinstance(parsed, dict):
        for key in ("queries", "questions", "tasks", "items"):
            if isinstance(parsed.get(key), list):
                return [_item_from_dict(o, default_id=f"{path.stem}_{i:02d}")
                        for i, o in enumerate(parsed[key], 1) if isinstance(o, dict)]
        return [_item_from_dict(parsed, default_id=path.stem)]
    if isinstance(parsed, list):
        return [_item_from_dict(o, default_id=f"{path.stem}_{i:02d}")
                for i, o in enumerate(parsed, 1) if isinstance(o, dict)]
    return []


def _extract_unknown_hints(unknown: List[str]) -> Tuple[Optional[str], Optional[str], Optional[Path], Optional[Path]]:
    """Extract positional query_id, query, question_file, or input_dir while ignoring unknown --flag pairs."""
    qid = query = qfile = idir = None
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("-"):
            if "=" not in tok and i + 1 < len(unknown) and not unknown[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        p = Path(tok)
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"} and qfile is None:
            qfile = p
        elif p.is_dir() and idir is None:
            idir = p
        elif _ID_RE.match(tok) and qid is None:
            qid = tok
        elif query is None:
            query = tok
        i += 1
    return qid, query, qfile, idir


def _stem_from_output(output_file: Optional[Path]) -> Optional[str]:
    if output_file is None or (output_file.exists() and output_file.is_dir()) or str(output_file).endswith(os.sep):
        return None
    stem = output_file.stem
    return stem[:-7] if stem.endswith("_output") else stem


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--input-image", type=Path, help="Match 2: read the text in this image")
    ap.add_argument("--index", nargs="?", const=Path("/app/corpus"), type=Path,
                    help="Match 3: build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="Match 3: corpus root for a query")
    ap.add_argument("--query-id", "--id", "--qid", "--find-id", "--find", "--task-id",
                    dest="query_id", help="output stem the harness assigns")
    ap.add_argument("--query", "--question", "--task", "--prompt", "-q", "--q",
                    dest="query", help="the question to answer")
    ap.add_argument("--url", "--website", "--start-url", dest="url", help="Match 4: optional start page")
    ap.add_argument("--pages", type=Path, help="Match 4: a folder of saved pages instead of the live web")
    ap.add_argument("--question-file", "--input", "--input-file", dest="question_file", type=Path,
                    help="Match 4: JSON/JSONL file with question fields")
    ap.add_argument("--input-dir", type=Path, help="Match 4: folder of JSON/JSONL question files")
    ap.add_argument("--output-dir", type=Path, help="Override MC3_OUTPUT_DIR for this call")
    ap.add_argument("--output", "-o", dest="output_file", type=Path, help="Explicit output JSON path")
    args, unknown = ap.parse_known_args()

    out_dir = args.output_dir or OUTPUT_DIR
    out_file = args.output_file

    if args.input_image is not None:
        run_ocr(args.input_image, output_dir=out_dir, output_file=out_file)
        return 0
    if args.index is not None:
        run_index(args.index)
        return 0

    u_qid, u_query, u_qfile, u_idir = _extract_unknown_hints(unknown)
    args.query_id = args.query_id or u_qid
    args.query = args.query or u_query
    args.question_file = args.question_file or u_qfile
    args.input_dir = args.input_dir or u_idir

    items: List[Dict[str, Optional[str]]] = []
    if args.input_dir and args.input_dir.is_dir():
        for fp in sorted(list(args.input_dir.glob("*.json")) + list(args.input_dir.glob("*.jsonl"))):
            try:
                items.extend(_load_file_items(fp))
            except Exception as err:
                print(f"unreadable input {fp}: {err}", file=sys.stderr)
    elif args.question_file:
        try:
            items = _load_file_items(args.question_file)
        except Exception as err:
            print(f"unreadable --question-file: {err}", file=sys.stderr)

    if not items:
        items = [{"query_id": args.query_id, "query": args.query, "url": args.url,
                  "pages": str(args.pages) if args.pages else None}]

    batch = len(items) > 1
    for idx, item in enumerate(items, 1):
        qid = (args.query_id if not batch else None) or item["query_id"] or _stem_from_output(out_file) or (
            f"query_{idx:02d}" if batch else "query"
        )
        q = (args.query if not batch else None) or item["query"]
        url = (args.url if not batch else None) or item["url"]
        pages = args.pages or (Path(item["pages"]) if item.get("pages") else None)
        target_file = None if batch else out_file

        if not q:
            write_json(qid, {"answer": "", "citations": [], "confidence": 0.0},
                       output_dir=out_dir, output_file=target_file)
            continue
        if args.corpus is not None and url is None and os.environ.get("MATCH_MODE") != "web":
            run_query(args.corpus, qid, q, output_dir=out_dir, output_file=target_file)
            continue
        run_web(qid, q, url, pages, output_dir=out_dir, output_file=target_file)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # the answer is written; don't let slow fetch threads hold the process past 30 s


if __name__ == "__main__":
    raise SystemExit(main())
