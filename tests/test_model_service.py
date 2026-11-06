"""The resident model service, the Qwen reader, image reading at index time and the Match 2 mode.

The real model only runs on an AMD GPU, so these tests use a stand-in model behind the
same unix-socket protocol the container uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from engine.client import ModelClient, ModelError
from engine.index import build_index, search
from engine.ocr import clean_ocr_text
from engine.readers import QwenReader, parse_reply
from engine.server import serve

ROOT = Path(__file__).resolve().parent.parent


class FakeModel:
    """Answers by rule, so each test controls exactly what 'the model' says."""

    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def ask(self, prompt, image_path=None, max_new_tokens=64):
        self.calls.append((prompt[:40], image_path))
        for key, reply in self.rules:
            if key in prompt or (image_path and key in image_path):
                if isinstance(reply, Exception):
                    raise reply
                return reply
        return ""


@pytest.fixture()
def socket_path():
    # Unix socket paths have a short length limit, so keep it out of pytest's deep tmp dirs.
    directory = tempfile.mkdtemp(prefix="mdl", dir="/tmp")
    yield os.path.join(directory, "m.sock")


def start_server(model, path):
    thread = threading.Thread(target=serve, args=(model, path), daemon=True)
    thread.start()
    ModelClient(path, wait_seconds=5).ping()
    return thread


# --- server and client ---------------------------------------------------

def test_client_round_trip(socket_path):
    start_server(FakeModel([("hello", "world")]), socket_path)
    assert ModelClient(socket_path).ask("hello there") == "world"


def test_model_errors_come_back_as_model_error(socket_path):
    start_server(FakeModel([("boom", RuntimeError("gpu fell over"))]), socket_path)
    with pytest.raises(ModelError, match="gpu fell over"):
        ModelClient(socket_path).ask("boom")
    assert ModelClient(socket_path).ask("still alive?") == ""  # server survives a bad request


def test_client_fails_fast_when_the_service_reported_a_failed_load(socket_path):
    Path(socket_path + ".failed").write_text("no GPU")
    started = time.monotonic()
    with pytest.raises(ModelError, match="failed to load"):
        ModelClient(socket_path, wait_seconds=30).ask("anyone?")
    assert time.monotonic() - started < 1


def test_client_gives_up_when_no_server(socket_path):
    started = time.monotonic()
    with pytest.raises(ModelError):
        ModelClient(socket_path, wait_seconds=0.5).ask("anyone?")
    assert time.monotonic() - started < 3


# --- reader --------------------------------------------------------------

def test_parse_reply_handles_fences_and_junk():
    assert parse_reply('```json\n{"answer": "94", "files": ["a.pdf"]}\n```') == {"answer": "94", "files": ["a.pdf"]}
    assert parse_reply("I think it is 94") == {"answer": "", "files": []}
    assert parse_reply('{"answer": "94", "files": "a.pdf"}') == {"answer": "94", "files": ["a.pdf"]}


def test_qwen_reader_sends_context_and_parses(socket_path):
    model = FakeModel([("Question: What is the limit", '{"answer": "42", "files": ["c/notes.txt"]}')])
    start_server(model, socket_path)
    reader = QwenReader(ModelClient(socket_path))
    reply = reader.read("What is the limit?", [{"file": "c/notes.txt", "location": "lines 1-1", "text": "limit is 42"}])
    assert reply == {"answer": "42", "files": ["c/notes.txt"]}


# --- images at index time ------------------------------------------------

def test_images_become_searchable_chunks(sample_corpus):
    def transcribe(path):
        if path.endswith("backplane_pinout.png"):
            return "TQ-40 backplane connector - pin assignment\nB14\nTHERM_ALERT#"
        raise RuntimeError("unreadable image")

    index = build_index(sample_corpus, image_reader=transcribe)
    image_chunks = [c for c in index["chunks"] if c["file"] == "specs/backplane_pinout.png"]
    assert image_chunks and "THERM_ALERT#" in image_chunks[0]["text"]
    assert {"file": "support/asset_label.jpg", "reason": "vision_error: RuntimeError"} in index["skipped"]
    top = search(index, "Which backplane pin carries THERM_ALERT# on the TQ-40?", k=1)
    assert top[0]["file"] == "specs/backplane_pinout.png"


# --- Match 2 text cleanup ------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("7ABC123", "7ABC123"),
    ('"7ABC123"', "7ABC123"),
    ("Text: 7ABC123", "7ABC123"),
    ("STOP\nAHEAD", "STOP AHEAD"),
    ("  35 \n", "35"),
    ("京A·12345", "京A·12345"),
])
def test_clean_ocr_text(raw, expected):
    assert clean_ocr_text(raw) == expected


# --- app.py end to end against the stand-in model --------------------------

def _run_app(socket_path, tmp_path, *args):
    env = {
        "PATH": "/usr/bin:/bin",
        "MODEL_SOCKET": socket_path,
        "MC3_READER": "qwen",
        "MC3_INDEX_DIR": str(tmp_path / "index"),
        "MC3_OUTPUT_DIR": str(tmp_path / "out"),
        "MODEL_WAIT_SECONDS": "5",
    }
    return subprocess.run([sys.executable, str(ROOT / "app.py"), *args], env=env,
                          capture_output=True, text=True, timeout=120)


def test_app_match2_writes_text_named_after_the_image(socket_path, tmp_path):
    start_server(FakeModel([("plate.png", '"7ABC123"\n')]), socket_path)
    image = tmp_path / "plate.png"
    image.write_bytes(b"not really a png; the stand-in model never opens it")
    done = _run_app(socket_path, tmp_path, "--input-image", str(image))
    assert done.returncode == 0, done.stderr
    out = json.loads((tmp_path / "out" / "plate_output.json").read_text())
    assert out["text"] == "7ABC123" and "confidence" in out


def test_app_match2_without_a_model_still_writes_valid_json(tmp_path):
    image = tmp_path / "sign.jpg"
    image.write_bytes(b"x")
    done = _run_app("/tmp/no-such-socket-here", tmp_path, "--input-image", str(image))
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "out" / "sign_output.json").read_text())["text"] == ""


def test_app_match3_query_through_the_model(sample_corpus, socket_path, tmp_path):
    rules = [
        ("Transcribe", "unused"),
        ("Question: Which firmware version fixed ticket ORR-1847?",
         '{"answer": "4.3.2", "files": ["support/bug_database.csv"]}'),
    ]
    start_server(FakeModel(rules), socket_path)
    assert _run_app(socket_path, tmp_path, "--index", str(sample_corpus)).returncode == 0
    done = _run_app(socket_path, tmp_path, "--corpus", str(sample_corpus), "--query-id", "query_04",
                    "--query", "Which firmware version fixed ticket ORR-1847?")
    assert done.returncode == 0, done.stderr
    out = json.loads((tmp_path / "out" / "query_04_output.json").read_text())
    assert (out["answer"], out["citations"]) == ("4.3.2", ["support/bug_database.csv"])
