"""Reader backends: the part that turns a question plus context into a value.

  none  refuses everything (pipeline tests without a model)
  qwen  Qwen3-VL-8B behind the resident model service; the prompt is the one that
        scored 200/200 on the Match 3 sample (notes/notebook_2026-11-06_qwen_reader.py)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from engine.client import ModelClient

CONTEXT_CHARS_PER_CHUNK = 1500
ANSWER_TOKENS = 80
TRANSCRIBE_PROMPT = "Transcribe all text in this image, line by line."
TRANSCRIBE_TOKENS = 300

PROMPT = """You answer a question using ONLY the numbered excerpts below. The products in them are fictional, so never use outside knowledge.
Reply with one line of JSON and nothing else: {{"answer": "...", "files": ["..."]}}
- answer: the exact value only, copied as written in the excerpt (a number, code, version, part number, quarter). Keep qualifiers that are part of the value, such as "Q3 FY27" or "REV-C2". No sentence, no units phrase, no explanation.
- files: the file paths you needed. If one excerpt gave an identifier (a ticket or error code) and another gave the value for it, list both. Never list a file you did not need.
- If a document says it is withdrawn or superseded, use the current one.
- If the excerpts do not contain the answer, reply {{"answer": "", "files": []}}.

Question: {q}

Excerpts:
{ctx}"""

REFUSAL = {"answer": "", "files": []}


def parse_reply(raw: str) -> Dict:
    """Pull the JSON object out of a model reply; anything unreadable is a refusal."""
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return dict(REFUSAL)
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return dict(REFUSAL)
    if not isinstance(data, dict):
        return dict(REFUSAL)
    files = data.get("files", [])
    files = [files] if isinstance(files, str) else [str(f) for f in files if isinstance(f, (str, int))]
    return {"answer": str(data.get("answer", "") or ""), "files": files}


def format_context(context: List[Dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] file: {c['file']} ({c['location']})\n{c['text'][:CONTEXT_CHARS_PER_CHUNK]}"
        for i, c in enumerate(context)
    )


class NoneReader:
    """Refuses every question. Lets the whole pipeline run before a model exists."""

    def read(self, question: str, context: List[Dict]) -> Dict:
        return {"answer": "", "files": [], "confidence": 0.0}


class QwenReader:
    def __init__(self, client: Optional[ModelClient] = None):
        self.client = client or ModelClient()

    def read(self, question: str, context: List[Dict]) -> Dict:
        raw = self.client.ask(PROMPT.format(q=question, ctx=format_context(context)), None, ANSWER_TOKENS)
        return parse_reply(raw)


def image_transcriber(client: Optional[ModelClient] = None):
    """A callable for build_index(image_reader=...) that reads each image once."""
    client = client or ModelClient()
    return lambda path: client.ask(TRANSCRIBE_PROMPT, path, TRANSCRIBE_TOKENS)


READERS = {"none": NoneReader, "qwen": QwenReader}


def get_reader(name: str):
    try:
        return READERS[name]()
    except KeyError:
        raise ValueError(f"unknown reader {name!r}; choose from {sorted(READERS)}") from None
