"""From a question to a checked answer with the smallest set of citations.

1. build_context: keyword search, minus withdrawn documents that have a current
   version, plus one "hop" that follows identifiers (ticket numbers, error codes)
   found in the first hits into the files that explain them.
2. The reader (the model) picks a value and the files it used.
3. check_answer: the value must be written in one of the cited files, and only
   retrieved files may be cited. Anything else becomes a refusal, because a wrong
   answer and a refusal score the same, but an honest refusal also wins the
   unanswerable questions.
"""

from __future__ import annotations

import re
import sys
from pathlib import PurePosixPath
from typing import Dict, List, Protocol, Tuple

from engine.index import search

CONTEXT_CHUNKS = 10
FIRST_HITS = 6
HOP_SOURCES = 3
_WITHDRAWN_NAME = re.compile(r"withdrawn|superseded|obsolete|deprecated", re.I)
_WITHDRAWN_TEXT = re.compile(r"\bwithdrawn\b|\bsuperseded by\b|\bobsolete\b", re.I)
_REVISION = re.compile(r"(^|_)(r|rev|v)\d+(?=_|$)", re.I)
_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]*-\d{2,}[A-Z0-9]*\b|\b[A-Z]\d{3,}\b")
_LOCATION_SUFFIX = re.compile(r"\s+\([^()]*\)\s*$")
_PROSE_REFUSAL = re.compile(r"don'?t know|not (found|available|mentioned|stated|provided)|no answer|unknown|cannot|n/?a\b", re.I)


class Reader(Protocol):
    def read(self, question: str, context: List[Dict]) -> Dict: ...


def normalize(value: str) -> str:
    """The grader's rule (uppercase, no whitespace, no - . · _), plus degree signs."""
    return re.sub(r"[\s\-.·_°]", "", value.upper())


def _family(rel: str) -> str:
    stem = PurePosixPath(rel).stem.lower()
    stem = _WITHDRAWN_NAME.sub("", stem)
    stem = _REVISION.sub(r"\1", stem)
    return str(PurePosixPath(rel).parent / re.sub(r"_+", "_", stem).strip("_"))


def withdrawn_files(index: Dict) -> set:
    """Files marked withdrawn (by name or text) that have a current file of the same family."""
    files: Dict[str, str] = {}
    for chunk in index["chunks"]:
        files.setdefault(chunk["file"], "")
        files[chunk["file"]] += " " + chunk["text"][:400]
    marked = {f for f, text in files.items() if _WITHDRAWN_NAME.search(f) or _WITHDRAWN_TEXT.search(text)}
    current_families = {_family(f) for f in files if f not in marked}
    return {f for f in marked if _family(f) in current_families}


def build_context(index: Dict, query: str) -> List[Dict]:
    dropped = withdrawn_files(index)
    hits = [h for h in search(index, query, k=FIRST_HITS + len(dropped) * 2) if h["file"] not in dropped][:FIRST_HITS]

    known = set(_IDENTIFIER.findall(query))
    hop_ids = []
    for hit in hits[:HOP_SOURCES]:
        for ident in _IDENTIFIER.findall(hit["text"]):
            if ident not in known and ident not in hop_ids:
                hop_ids.append(ident)
    for ident in hop_ids:
        hits += [h for h in search(index, ident, k=3) if ident in h["text"] and h["file"] not in dropped]

    context, seen = [], set()
    for hit in hits:
        key = (hit["file"], hit["location"])
        if key not in seen:
            seen.add(key)
            context.append(hit)
    return context[:CONTEXT_CHUNKS]


def check_answer(answer: str, files: List[str], context: List[Dict]) -> Tuple[str, List[str]]:
    value = (answer or "").strip()
    if not value or _PROSE_REFUSAL.search(value):
        return "", []
    retrieved = {c["file"] for c in context}
    # Models copy the context label "file (location)"; map that back to the file it names.
    files = [f if f in retrieved else _LOCATION_SUFFIX.sub("", f) for f in (files or [])]
    cited = [f for f in dict.fromkeys(files) if f in retrieved]
    wanted = normalize(value)
    holders = {c["file"] for c in context if c["file"] in cited and wanted and wanted in normalize(c["text"])}
    if not holders:
        return "", []
    # Necessity, not relevance: keep a file only if it holds the answer, or shares an
    # identifier (ticket, error code) with an answer-holding chunk, i.e. it is a step of the chain.
    chain_ids = {i for c in context if c["file"] in holders and wanted in normalize(c["text"])
                 for i in _IDENTIFIER.findall(c["text"])}
    linked = {c["file"] for c in context if c["file"] in cited and chain_ids & set(_IDENTIFIER.findall(c["text"]))}
    return value, [f for f in cited if f in holders or f in linked]


def answer_question(index: Dict, query: str, reader: Reader) -> Tuple[str, List[str], float]:
    context = build_context(index, query)
    if not context:
        return "", [], 0.0
    try:
        reply = reader.read(query, context)
    except Exception as err:  # a crashed model must still produce a valid file
        print(f"reader failed: {err.__class__.__name__}: {err}", file=sys.stderr)
        return "", [], 0.0
    value, cited = check_answer(str(reply.get("answer", "")), list(reply.get("files", [])), context)
    confidence = float(reply.get("confidence", 0.5)) if value else 0.0
    return value, cited, confidence
