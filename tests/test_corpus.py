"""Corpus walk, per-type parsing and the persisted search index."""

from __future__ import annotations

from pathlib import Path

from engine.index import build_index, load_index, search
from engine.parsers import parse_file
from engine.walker import walk_corpus


def _text(corpus: Path, rel: str) -> str:
    result = parse_file(corpus / rel)
    assert result.status == "ok", f"{rel}: {result.status} {result.reason}"
    return "\n".join(seg.text for seg in result.segments)


# --- walk ---------------------------------------------------------------

def test_walk_lists_every_file_in_sorted_order(sample_corpus):
    files = walk_corpus(sample_corpus)
    rels = [f.as_posix() for f in files]
    assert rels == sorted(rels)
    assert len(rels) == 13  # the kit says 13 files across 6 subdirectories with files
    assert "vendor/internal_audit.txt" in rels


def test_unreadable_first_file_does_not_stop_the_walk(hostile_corpus):
    index = build_index(hostile_corpus)
    indexed = {c["file"] for c in index["chunks"]}
    assert "c_data/notes.txt" in indexed
    reasons = {s["file"]: s["reason"] for s in index["skipped"]}
    assert reasons["a_first/aaa_unreadable.txt"] == "unreadable"
    assert reasons["c_data/blob.xyz"] == "unknown_type"


# --- parsers: one sample answer per file type ----------------------------

def test_pdf(sample_corpus):
    assert "94" in _text(sample_corpus, "specs/tq40_datasheet_r2.pdf")


def test_docx_reads_tables(sample_corpus):
    assert "Q3 FY27" in _text(sample_corpus, "planning/roadmap_fy27.docx")


def test_xlsx(sample_corpus):
    assert "ORR-FAN-2214-B" in _text(sample_corpus, "support/rma_parts.xlsx")


def test_csv_keeps_ticket_and_fix_in_one_row(sample_corpus):
    result = parse_file(sample_corpus / "support/bug_database.csv")
    rows = [s.text for s in result.segments if "ORR-1847" in s.text]
    assert rows and "4.3.2" in rows[0]


def test_log(sample_corpus):
    assert "E7731" in _text(sample_corpus, "logs/prod_inference_2026-09-02.log")


def test_python_source(sample_corpus):
    assert "180" in _text(sample_corpus, "engineering/ingest_service.py")


def test_images_are_queued_for_the_vision_model(sample_corpus):
    for rel in ("specs/backplane_pinout.png", "support/asset_label.jpg"):
        assert parse_file(sample_corpus / rel).status == "needs_vision"


def test_trap_files_are_skipped_with_a_reason(sample_corpus):
    assert parse_file(sample_corpus / "vendor/internal_audit.txt").reason == "unreadable"
    assert parse_file(sample_corpus / "vendor/supplier_agreement_ENCRYPTED.pdf").reason == "encrypted"
    assert parse_file(sample_corpus / "vendor/telemetry_capture.dat").reason == "unknown_type"


# --- index --------------------------------------------------------------

def test_index_round_trips_and_accounts_for_every_file(sample_corpus, tmp_path):
    built = build_index(sample_corpus, tmp_path)
    loaded = load_index(tmp_path)
    assert loaded == built
    accounted = {c["file"] for c in built["chunks"]} | {s["file"] for s in built["skipped"]} | set(built["images"])
    assert len(accounted) == 13


def test_search_ranks_the_right_file_first(sample_corpus):
    index = build_index(sample_corpus)
    cases = {
        "What is the part number of the field-replaceable fan assembly for the TQ-40?": "support/rma_parts.xlsx",
        "Which firmware version fixed ticket ORR-1847?": "support/bug_database.csv",
        "What error code is logged when the thermal throttle engages?": "logs/prod_inference_2026-09-02.log",
        "What is the default batch timeout, in seconds, in the ingest service?": "engineering/ingest_service.py",
        "In which quarter does the TQ-60 enter customer sampling?": "planning/roadmap_fy27.docx",
    }
    for query, expected in cases.items():
        top = search(index, query, k=3)
        assert top and top[0]["file"] == expected, f"{query!r} -> {[h['file'] for h in top]}"
