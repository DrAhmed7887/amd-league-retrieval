"""Reader backends: the part that turns a question plus context into a value.

Only "none" exists so far. The real reader (an open model resident on the GPU,
reached over a unix socket so it loads once, not once per question) comes next.
"""

from __future__ import annotations

from typing import Dict, List


class NoneReader:
    """Refuses every question. Lets the whole pipeline run before a model exists."""

    def read(self, question: str, context: List[Dict]) -> Dict:
        return {"answer": "", "files": [], "confidence": 0.0}


READERS = {"none": NoneReader}


def get_reader(name: str):
    try:
        return READERS[name]()
    except KeyError:
        raise ValueError(f"unknown reader {name!r}; choose from {sorted(READERS)}") from None
