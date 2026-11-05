"""Shared fixtures. The sample corpus comes from AMD's Match 3 starter kit (kit/, not in git)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CORPUS = ROOT / "kit" / "mc3-corpus"


@pytest.fixture(scope="session")
def sample_corpus() -> Path:
    if not SAMPLE_CORPUS.is_dir():
        pytest.skip("starter kit not unpacked at kit/mc3-corpus")
    unreadable = SAMPLE_CORPUS / "vendor" / "internal_audit.txt"
    if os.access(unreadable, os.R_OK):
        pytest.skip("run kit/setup.sh first: internal_audit.txt must be mode 000")
    return SAMPLE_CORPUS


@pytest.fixture()
def hostile_corpus(tmp_path: Path) -> Path:
    """A corpus whose FIRST entry in listing order is unreadable, to prove the walk keeps going."""
    (tmp_path / "a_first").mkdir()
    bad = tmp_path / "a_first" / "aaa_unreadable.txt"
    bad.write_text("secret value 999", encoding="utf-8")
    bad.chmod(0)
    (tmp_path / "b_empty").mkdir()
    (tmp_path / "c_data").mkdir()
    (tmp_path / "c_data" / "notes.txt").write_text("The widget limit is 42 units.", encoding="utf-8")
    (tmp_path / "c_data" / "blob.xyz").write_bytes(b"\x00\x01\x02")
    yield tmp_path
    bad.chmod(stat.S_IRUSR | stat.S_IWUSR)
