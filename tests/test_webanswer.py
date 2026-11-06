"""Match 4 answer step with fake page sources and a fake reader: no network, no model."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from engine.webanswer import Sources, answer_web

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FakePage:
    url: str
    text: str
    final_url: str = ""
    status: int = 200
    title: str = ""
    links: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        self.final_url = self.final_url or self.url


def chunk(page):
    return [{"file": page.final_url, "location": "part 1", "text": page.text}]


PAGES = {
    "https://psf.example/about": FakePage("https://psf.example/about",
                                          "About the Python Software Foundation. The PSF was founded in 2001."),
    "https://psf.example/news": FakePage("https://psf.example/news", "News and events from the community."),
}


def fake_sources(searched=None, crawled=None):
    calls = {"search": 0, "crawl": []}

    def candidate_urls(query, deadline, k=8):
        calls["search"] += 1
        return list(searched if searched is not None else PAGES)

    def fetch(url, deadline):
        return PAGES.get(url)

    def crawl(url, query, deadline, max_pages=10, max_depth=2, same_site=True):
        calls["crawl"].append((url, max_pages, max_depth))
        return list(crawled if crawled is not None else [PAGES[u] for u in PAGES if u.startswith(url[:18])])

    return Sources(fetch=fetch, crawl=crawl, chunk_page=chunk, candidate_urls=candidate_urls,
                   load_saved_pages=lambda d: []), calls


class Reader:
    def __init__(self, answer, files):
        self.reply = {"answer": answer, "files": files}

    def read(self, question, context):
        return dict(self.reply)


Q = "Which year was the Python Software Foundation founded?"


def test_answer_from_search_cites_only_the_page_that_holds_the_value():
    sources, calls = fake_sources()
    value, cited, conf = answer_web(Q, Reader("2001", list(PAGES)), sources=sources)
    assert value == "2001"
    assert cited == ["https://psf.example/about"]
    assert conf == 0.9 and calls["search"] == 1


def test_value_not_on_any_cited_page_is_refused():
    sources, _ = fake_sources()
    assert answer_web(Q, Reader("1991", list(PAGES)), sources=sources) == ("", [], 0.0)


def test_prose_answer_is_refused():
    sources, _ = fake_sources()
    assert answer_web(Q, Reader("not found on the page", list(PAGES)), sources=sources)[0] == ""


def test_start_url_crawls_and_skips_search():
    sources, calls = fake_sources()
    value, cited, _ = answer_web(Q, Reader("2001", ["https://psf.example/about"]),
                                 url="https://psf.example/", sources=sources)
    assert value == "2001" and calls["search"] == 0
    assert calls["crawl"][0][0] == "https://psf.example/"


def test_no_pages_means_empty_answer():
    sources, _ = fake_sources(searched=[], crawled=[])
    assert answer_web(Q, Reader("2001", []), sources=sources) == ("", [], 0.0)


def test_exhausted_budget_refuses_without_asking_the_model():
    class Boom:
        def read(self, q, c):
            raise AssertionError("model must not be asked after the budget")
    sources, _ = fake_sources()
    assert answer_web(Q, Boom(), sources=sources, start=time.monotonic() - 60)[0] == ""


def test_crashing_source_still_returns_a_refusal():
    sources, _ = fake_sources()
    sources.candidate_urls = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("search down"))
    assert answer_web(Q, Reader("2001", []), sources=sources) == ("", [], 0.0)


def test_app_web_mode_writes_both_graded_keys(tmp_path):
    """No --corpus means Match 4 mode; with the 'none' reader it must still write a valid refusal fast."""
    env = {"MC3_OUTPUT_DIR": str(tmp_path), "MC3_READER": "none", "PATH": "/usr/bin:/bin",
           "PYTHONPATH": str(ROOT)}
    proc = subprocess.run([sys.executable, str(ROOT / "app.py"), "--id", "q01", "--question", "x?",
                           "--pages", str(tmp_path / "empty")], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads((tmp_path / "q01_output.json").read_text())
    assert out["answer"] == "" and out["citations"] == [] and "confidence" in out


def test_app_input_file_fills_the_question(tmp_path):
    spec = tmp_path / "q.json"
    spec.write_text(json.dumps({"id": "q02", "question": "x?"}))
    (tmp_path / "pages").mkdir()
    env = {"MC3_OUTPUT_DIR": str(tmp_path), "MC3_READER": "none", "PATH": "/usr/bin:/bin",
           "PYTHONPATH": str(ROOT)}
    proc = subprocess.run([sys.executable, str(ROOT / "app.py"), "--question-file", str(spec), "--pages", str(tmp_path / "pages")],
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "q02_output.json").exists()


def test_expand_query_adds_plurals_and_synonyms():
    from engine.webanswer import expand_query
    words = expand_query("What is the phone number for individual tax help?").split()
    assert {"telephone", "individuals", "phone"} <= set(words)


def test_web_context_includes_lead_passage_of_top_pages():
    from engine.webanswer import web_context
    chunks = [{"file": "https://a.example/x", "location": "part 1", "text": "Infobox Founded 6 March 2001"},
              {"file": "https://a.example/x", "location": "part 2", "text": "grants and programs funding"},
              {"file": "https://b.example/y", "location": "part 1", "text": "unrelated navigation menu"}]
    ctx = web_context(chunks, "programs funding")
    files_locs = [(c["file"], c["location"]) for c in ctx]
    assert files_locs[0] == ("https://a.example/x", "part 2")          # best keyword hit first
    assert ("https://a.example/x", "part 1") in files_locs              # lead passage always included
    assert ("https://b.example/y", "part 1") in files_locs
    assert len(files_locs) == len(set(files_locs))


def test_wikipedia_non_article_links_are_junk():
    from engine.web import JUNK
    assert JUNK.search("https://en.wikipedia.org/wiki/Talk:Python_Software_Foundation")
    assert JUNK.search("https://en.wikipedia.org/wiki/Special:WhatLinksHere/X")
    assert not JUNK.search("https://en.wikipedia.org/wiki/Python_Software_Foundation")
