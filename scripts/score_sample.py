#!/usr/bin/env python3
"""Score the starter kit's 10 sample questions the way the grader does.

A question scores 20 only if the normalized answer matches AND the citations are
exactly the expected set. Readers:
  none    refuses everything (should score only the unanswerable question)
  oracle  answers with the expected value and files, so the score is the ceiling
          our retrieval and checks allow: it fails where the right file never
          reached the context, or where a check rejects a right answer.

Run: .venv/bin/python scripts/score_sample.py [--reader none|oracle]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.answer import answer_question, normalize  # noqa: E402
from engine.index import build_index  # noqa: E402
from engine.readers import get_reader  # noqa: E402

KIT = ROOT / "kit"


class OracleReader:
    def __init__(self):
        self.expected = None

    def read(self, question, context):
        return {"answer": self.expected["expected_answer"], "files": self.expected["expected_citations"]}


def is_correct(answer, citations, q) -> tuple:
    accepted = {normalize(a) for a in [q["expected_answer"], *q.get("answer_aliases", [])]}
    answer_ok = normalize(answer) in accepted if answer else q["expected_answer"] == ""
    cites_ok = set(citations) == set(q["expected_citations"])
    return answer_ok, cites_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="oracle")
    args = ap.parse_args()
    questions = json.loads((KIT / "sample-questions.json").read_text())["queries"]
    index = build_index(KIT / "mc3-corpus")
    reader = OracleReader() if args.reader == "oracle" else get_reader(args.reader)

    total = 0
    print(f"reader: {args.reader}\n")
    print(f"{'#':>2}  {'score':>5}  {'answer':<16} {'citations':<55} note")
    for q in questions:
        if isinstance(reader, OracleReader):
            reader.expected = q
        answer, citations, _ = answer_question(index, q["query"], reader)
        answer_ok, cites_ok = is_correct(answer, citations, q)
        score = 20 if answer_ok and cites_ok else 0
        total += score
        note = "" if score else ("wrong answer" if not answer_ok else "wrong citations")
        print(f"{q['n']:>2}  {score:>5}  {answer or '(empty)':<16} {', '.join(citations) or '(none)':<55} {note}")
    print(f"\ntotal {total}/200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
