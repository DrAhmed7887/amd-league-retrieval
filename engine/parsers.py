"""Turn one corpus file into searchable text segments.

Each segment keeps a location (page, sheet and row, line range) so an answer can
later be traced to where it came from. Parsing never raises: a file that cannot
be used comes back with a status and a reason instead.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

TEXT_TYPES = {".txt", ".log", ".md", ".py", ".json", ".yaml", ".yml", ".ini", ".cfg", ".toml", ".sh", ".xml", ".html", ".htm"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
LINES_PER_SEGMENT = 20


@dataclass(frozen=True)
class Segment:
    location: str
    text: str


@dataclass(frozen=True)
class ParseResult:
    status: str  # "ok" | "needs_vision" | "skipped"
    segments: Tuple[Segment, ...] = field(default_factory=tuple)
    reason: str = ""


def _skip(reason: str) -> ParseResult:
    return ParseResult(status="skipped", reason=reason)


def _ok(segments: List[Segment]) -> ParseResult:
    kept = tuple(s for s in segments if s.text.strip())
    return ParseResult(status="ok", segments=kept) if kept else _skip("empty")


def _parse_text(data: bytes) -> List[Segment]:
    lines = data.decode("utf-8", errors="replace").splitlines()
    segments = []
    for start in range(0, len(lines), LINES_PER_SEGMENT):
        chunk = lines[start:start + LINES_PER_SEGMENT]
        segments.append(Segment(f"lines {start + 1}-{start + len(chunk)}", "\n".join(chunk)))
    return segments


def _rows_as_text(header: List[str], row: List[str]) -> str:
    if header and len(header) == len(row):
        return "; ".join(f"{h}: {v}" for h, v in zip(header, row) if str(v).strip())
    return " | ".join(str(v) for v in row if str(v).strip())


def _parse_csv(data: bytes) -> List[Segment]:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8", errors="replace"))))
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    segments = [Segment("header", " | ".join(header))]
    for n, row in enumerate(rows[1:], start=2):
        segments.append(Segment(f"row {n}", _rows_as_text(header, row)))
    return segments


def _parse_pdf(path: Path) -> ParseResult:
    from pypdf import PdfReader

    # Any encrypted PDF is out of bounds, even one that opens with an empty
    # password: the brief scores its contents as "not a source". Check the raw
    # bytes first, so no decryption is ever attempted.
    if b"/Encrypt" in path.read_bytes():
        return _skip("encrypted")
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        return _skip("encrypted")
    segments = [Segment(f"page {i}", page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    return _ok(segments)


def _parse_docx(path: Path) -> List[Segment]:
    import docx

    document = docx.Document(str(path))
    segments = [Segment(f"paragraph {i}", p.text) for i, p in enumerate(document.paragraphs, start=1)]
    for t, table in enumerate(document.tables, start=1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        header = rows[0]
        segments.append(Segment(f"table {t} header", " | ".join(header)))
        for r, row in enumerate(rows[1:], start=2):
            segments.append(Segment(f"table {t} row {r}", _rows_as_text(header, row)))
    return segments


def _parse_xlsx(path: Path) -> List[Segment]:
    import openpyxl

    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    segments = []
    try:
        for sheet in workbook.worksheets:
            rows = [["" if v is None else str(v) for v in row] for row in sheet.iter_rows(values_only=True)]
            rows = [r for r in rows if any(c.strip() for c in r)]
            if not rows:
                continue
            header = rows[0]
            segments.append(Segment(f"sheet {sheet.title} header", " | ".join(header)))
            for n, row in enumerate(rows[1:], start=2):
                segments.append(Segment(f"sheet {sheet.title} row {n}", f"[{sheet.title}] " + _rows_as_text(header, row)))
    finally:
        workbook.close()
    return segments


def parse_file(path: Path) -> ParseResult:
    """Parse one file. Never raises."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except PermissionError:
        return _skip("unreadable")
    except OSError as err:
        return _skip(f"unreadable: {err.__class__.__name__}")

    try:
        if suffix in IMAGE_TYPES:
            return ParseResult(status="needs_vision")
        if suffix == ".pdf":
            return _parse_pdf(path)
        if suffix == ".docx":
            return _ok(_parse_docx(path))
        if suffix in {".xlsx", ".xlsm"}:
            return _ok(_parse_xlsx(path))
        if suffix in {".csv", ".tsv"}:
            return _ok(_parse_csv(path.read_bytes()))
        if suffix in TEXT_TYPES:
            return _ok(_parse_text(path.read_bytes()))
        return _skip("unknown_type")
    except Exception as err:  # one broken file must never stop the corpus
        return _skip(f"parse_error: {err.__class__.__name__}: {err}"[:200])
