"""Context building (withdrawn documents, identifier hops), answer checks and the app contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from engine.answer import answer_question, build_context, check_answer, normalize
from engine.index import build_index

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def index(sample_corpus):
    return build_index(sample_corpus)


def _files(context):
    return {c["file"] for c in context}


class ScriptedReader:
    """Stands in for the model: returns a fixed reply and records what it was shown."""

    def __init__(self, answer, files):
        self.reply = {"answer": answer, "files": files}
        self.seen = None

    def read(self, question, context):
        self.seen = context
        return dict(self.reply)


class BrokenReader:
    def read(self, question, context):
        raise RuntimeError("model crashed")


# --- context -------------------------------------------------------------

def test_withdrawn_revision_is_dropped_when_a_current_one_exists(index):
    context = build_context(index, "What is the maximum junction temperature of the TQ-40?")
    files = _files(context)
    assert "specs/tq40_datasheet_r2.pdf" in files
    assert "specs/tq40_datasheet_r1_WITHDRAWN.pdf" not in files


def test_identifier_hop_reaches_the_second_file(index):
    query = "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
    context = build_context(index, query)
    assert "logs/prod_inference_2026-09-02.log" in _files(context)
    assert any("ORR-1847" in c["text"] and "4.3.2" in c["text"] for c in context if c["file"] == "support/bug_database.csv")


# --- answer checks -------------------------------------------------------

def test_normalize_matches_the_grader_rule():
    assert normalize("7-ABC 1.23") == "7ABC123"
    assert normalize("Q3 FY27") == normalize("q3fy27")


def test_check_accepts_a_supported_answer():
    context = [{"file": "a.csv", "text": "ticket: ORR-1847; fixed_in: 4.3.2"}]
    assert check_answer("4.3.2", ["a.csv"], context) == ("4.3.2", ["a.csv"])


def test_check_refuses_an_answer_missing_from_the_cited_files():
    context = [{"file": "a.csv", "text": "fixed_in: 4.3.2"}, {"file": "b.txt", "text": "4.3.3 packaging"}]
    assert check_answer("4.3.9", ["a.csv"], context) == ("", [])
    assert check_answer("4.3.2", ["b.txt"], context) == ("", [])


def test_check_drops_citations_that_were_never_retrieved():
    context = [{"file": "a.csv", "text": "fixed_in: 4.3.2"}]
    assert check_answer("4.3.2", ["a.csv", "nowhere.pdf"], context) == ("4.3.2", ["a.csv"])


def test_over_citation_is_pruned_to_files_on_the_chain():
    context = [
        {"file": "support/bugs.csv", "text": "ticket: ORR-1847; fixed_in: 4.3.2"},
        {"file": "logs/prod.log", "text": "ERROR E7731 throttle; incident logged against ORR-1847"},
        {"file": "engineering/notes.txt", "text": "4.3.1 fixes a doorbell hang"},
    ]
    cited = ["logs/prod.log", "support/bugs.csv", "engineering/notes.txt"]
    assert check_answer("4.3.2", cited, context) == ("4.3.2", ["logs/prod.log", "support/bugs.csv"])


def test_empty_answer_means_no_citations():
    assert check_answer("", ["a.csv"], [{"file": "a.csv", "text": "x"}]) == ("", [])


def test_prose_refusals_become_real_refusals():
    context = [{"file": "a.csv", "text": "I don't know"}]
    assert check_answer("I don't know", ["a.csv"], context) == ("", [])


# --- end to end with a stand-in model ------------------------------------

def test_answer_question_runs_the_checks(index):
    reader = ScriptedReader("94", ["specs/tq40_datasheet_r2.pdf"])
    answer, citations, _ = answer_question(index, "What is the maximum junction temperature of the TQ-40?", reader)
    assert (answer, citations) == ("94", ["specs/tq40_datasheet_r2.pdf"])
    assert "specs/tq40_datasheet_r1_WITHDRAWN.pdf" not in _files(reader.seen)


def test_a_crashing_model_becomes_a_refusal(index):
    assert answer_question(index, "anything", BrokenReader())[:2] == ("", [])


def test_app_writes_the_contract_file(sample_corpus, tmp_path):
    env = {"MC3_INDEX_DIR": str(tmp_path / "index"), "MC3_OUTPUT_DIR": str(tmp_path / "out"), "MC3_READER": "none"}
    run = lambda *args: subprocess.run([sys.executable, str(ROOT / "app.py"), *args], env={**env, "PATH": "/usr/bin:/bin"},
                                       capture_output=True, text=True, timeout=120)
    assert run("--index", str(sample_corpus)).returncode == 0
    done = run("--corpus", str(sample_corpus), "--query-id", "query_07", "--query", "What is the price?")
    assert done.returncode == 0, done.stderr
    out = json.loads((tmp_path / "out" / "query_07_output.json").read_text())
    assert set(out) == {"answer", "citations", "confidence"}
    assert out["answer"] == "" and out["citations"] == []
