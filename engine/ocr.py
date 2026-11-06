"""Match 2: read the text on a licence plate or road sign.

The grader normalises answers (uppercase, no whitespace, no - . · _), so the job here
is to return only the characters that are graded: the plate number or the sign's
message, in reading order, lines joined with one space, no labels or units.
"""

from __future__ import annotations

import re

OCR_PROMPT = """Read the text on this licence plate or road sign.
Reply with only the characters that are printed on it, in reading order, exactly as printed.
- For a licence plate, give only the registration number. Leave out text printed around it, such as a state or country name, a slogan, or a dealer frame.
- On a Chinese plate, keep the leading province character and letter: they are part of the number (for example 京A·12345).
- If the text runs over several lines, read top to bottom and join the lines with one space.
- Do not add units, labels, quotes or any explanation."""

_LABEL = re.compile(r"^\s*(text|answer|plate|plate number|the text is)\s*[:：]\s*", re.I)


def clean_ocr_text(raw: str) -> str:
    text = _LABEL.sub("", (raw or "").strip())
    text = text.strip().strip('"\'`“”‘’').strip()
    return re.sub(r"\s+", " ", text)
