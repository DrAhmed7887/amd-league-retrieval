"""Build, persist and search the corpus index.

The index is a plain JSON file written once during --index, then reloaded by each
per-question process. Search is BM25 over text segments, with each file's path
words added to its segments so "ingest service" finds engineering/ingest_service.py.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

from engine.parsers import parse_file
from engine.walker import walk_corpus

INDEX_FILE = "index.json"
BM25_K1 = 1.5
BM25_B = 0.75
STOPWORDS = frozenset(
    "a an and are as at be by does for from has have in is it of on or the to was what when which who why with "
    "how many much this that these those there their its into than then".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.#][a-z0-9]+)*#?")


def tokenize(text: str) -> List[str]:
    """Lowercase tokens. Compound tokens like orr-1847 are kept whole and also split."""
    tokens: List[str] = []
    for tok in _TOKEN.findall(text.lower()):
        parts = [p for p in re.split(r"[-_.#]", tok) if p]
        for t in ([tok] if len(parts) > 1 else []) + parts:
            if t not in STOPWORDS:
                tokens.append(t)
    return tokens


def _path_words(rel: str) -> str:
    return " ".join(re.split(r"[/_.\-]", rel))


def build_index(corpus: Path, index_dir: Optional[Path] = None,
                image_reader: Optional[Callable[[str], str]] = None) -> Dict:
    """Parse every file once. With image_reader, each image is transcribed once and
    its text indexed like any other chunk; a failed transcription is skipped with a reason."""
    corpus = Path(corpus)
    chunks: List[Dict] = []
    skipped: List[Dict] = []
    images: List[str] = []
    for rel_path in walk_corpus(corpus):
        rel = rel_path.as_posix()
        result = parse_file(corpus / rel_path)
        if result.status == "needs_vision":
            images.append(rel)
            if image_reader is not None:
                try:
                    text = image_reader(str(corpus / rel_path))
                except Exception as err:
                    skipped.append({"file": rel, "reason": f"vision_error: {err.__class__.__name__}"})
                    continue
                if text.strip():
                    chunks.append({"file": rel, "location": "image text", "text": text})
        elif result.status == "skipped":
            skipped.append({"file": rel, "reason": result.reason})
        else:
            for seg in result.segments:
                chunks.append({"file": rel, "location": seg.location, "text": seg.text})
    index = {"chunks": chunks, "skipped": skipped, "images": images}
    if index_dir is not None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp = index_dir / (INDEX_FILE + ".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        tmp.replace(index_dir / INDEX_FILE)
    return index


def load_index(index_dir: Path) -> Dict:
    return json.loads((Path(index_dir) / INDEX_FILE).read_text(encoding="utf-8"))


def search(index: Dict, query: str, k: int = 5) -> List[Dict]:
    """Return the top-k chunks by BM25, best first, each with a 'score'."""
    docs = [tokenize(c["text"] + " " + _path_words(c["file"])) for c in index["chunks"]]
    if not docs:
        return []
    n_docs = len(docs)
    avg_len = sum(len(d) for d in docs) / n_docs
    doc_freq = Counter(t for d in docs for t in set(d))
    q_terms = set(tokenize(query))
    scored = []
    for chunk, doc in zip(index["chunks"], docs):
        tf = Counter(doc)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = tf[term] * (BM25_K1 + 1) / (tf[term] + BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avg_len))
            score += idf * norm
        if score > 0:
            scored.append({**chunk, "score": round(score, 4)})
    scored.sort(key=lambda c: (-c["score"], c["file"], c["location"]))
    return scored[:k]
