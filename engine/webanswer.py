"""Match 4: answer a question from web pages, citing the page(s) that hold the value.

Pages become chunks whose "file" is the page URL, so the Match 3 machinery is reused as is:
BM25 context, the reader, and check_answer (the value must be written in a cited page,
citations pruned to the pages the answer needed, refusal otherwise).

Time: the harness allows 30 s per question. Page gathering stops at FETCH_BUDGET so the
model always has time to answer inside TOTAL_BUDGET.
"""

from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from engine.answer import Reader, check_answer
from engine.index import search, tokenize

FETCH_BUDGET = 17.0     # seconds for search + fetch + crawl
SEARCH_BUDGET = 6.0     # the search engines alone may not eat the page-fetch time
TOTAL_BUDGET = 26.0     # leaves ~4 s of the harness's 30 s for process start and the write
SEARCH_CANDIDATES = 6
CRAWL_PAGES_WITH_URL = 10
CRAWL_PAGES_FROM_SEARCH = 3


CONTEXT_HITS = 9        # best keyword-matched passages
LEAD_PAGES = 3          # plus the opening passage of the top pages (infoboxes and summaries live there)
CONTEXT_MAX = 13

# Retrieval-only word expansion: pages say "telephone" and "Individuals" where questions say
# "phone" and "individual". The reader still sees the original question.
SYNONYMS = {
    "phone": "telephone call number contact", "telephone": "phone call", "email": "contact mail",
    "founded": "established formed founding", "established": "founded",
    "ceo": "chief executive officer president", "headquarters": "headquartered hq based located",
    "price": "cost costs pricing", "cost": "price pricing", "address": "located location street",
    "wrote": "author written writer", "author": "wrote written", "version": "release latest",
    "latest": "current newest release version", "born": "birth", "population": "inhabitants",
    "port": "default", "capital": "capital city",
}


# Titles the model keeps despite WEB_PROMPT ("Dr. Lisa Su"). Only dotted forms and "Professor",
# so brand names like "Dr Pepper" stay intact.
_HONORIFIC = re.compile(r"^(?:(?:Dr|Prof|Mr|Mrs|Ms|Mx)\.|Professor)\s+(?=[A-Z])")


def strip_honorific(value: str) -> str:
    return _HONORIFIC.sub("", value.strip())


def expand_query(query: str) -> str:
    words = tokenize(query)
    extra = []
    for w in words:
        extra.append(w[:-1] if w.endswith("s") and len(w) > 3 else w + "s")
        extra.extend(SYNONYMS.get(w, "").split())
    return " ".join(words + extra)


def web_context(chunks: List[Dict], query: str) -> List[Dict]:
    """Best keyword passages plus the lead passage of the top pages, de-duplicated, in that order."""
    hits = search({"chunks": chunks}, expand_query(query), k=CONTEXT_HITS)
    leads, seen_pages = [], []
    for c in chunks:  # chunks arrive in page order, best-ranked page first
        if c["file"] not in seen_pages:
            seen_pages.append(c["file"])
            leads.append(c)
            if len(leads) == LEAD_PAGES:
                break
    context, seen = [], set()
    for c in hits + leads:
        key = (c["file"], c["location"])
        if key not in seen:
            seen.add(key)
            context.append(c)
    return context[:CONTEXT_MAX]


class Sources:
    """The page-gathering functions, injectable so tests never touch the network."""

    def __init__(self, fetch=None, crawl=None, chunk_page=None, candidate_urls=None, load_saved_pages=None):
        if fetch is None or crawl is None or chunk_page is None or load_saved_pages is None:
            from engine import web
            fetch, crawl = fetch or web.fetch, crawl or web.crawl
            chunk_page, load_saved_pages = chunk_page or web.chunk_page, load_saved_pages or web.load_saved_pages
        if candidate_urls is None:
            from engine import websearch
            candidate_urls = websearch.candidate_urls
        self.fetch, self.crawl, self.chunk_page = fetch, crawl, chunk_page
        self.candidate_urls, self.load_saved_pages = candidate_urls, load_saved_pages


def gather_pages(query: str, url: Optional[str], corpus: Optional[Path], sources: Sources, deadline: float) -> List:
    """Saved pages if a folder is given; else crawl the start URL; else search, fetch and peek one level."""
    if corpus is not None and corpus.is_dir():
        return sources.load_saved_pages(corpus)
    if url:
        return sources.crawl(url, query, deadline, max_pages=CRAWL_PAGES_WITH_URL, max_depth=2, same_site=True)
    urls = sources.candidate_urls(query, min(deadline, time.monotonic() + SEARCH_BUDGET), k=SEARCH_CANDIDATES)
    pages = _fetch_all(urls, sources.fetch, deadline)
    if pages and time.monotonic() < deadline - 3:
        # One level deeper from the best hit: answers often sit one click away (contact, specs, releases).
        best = pages[0]
        extra = sources.crawl(best.final_url, query, deadline, max_pages=CRAWL_PAGES_FROM_SEARCH + 1,
                              max_depth=1, same_site=True)
        pages += [p for p in extra if p.final_url not in {q.final_url for q in pages}]
    return pages


def _fetch_all(urls: List[str], fetch: Callable, deadline: float) -> List:
    """Fetch concurrently; keep the search order so the best-ranked page stays first."""
    if not urls:
        return []
    results: Dict[int, object] = {}
    pool = ThreadPoolExecutor(max_workers=min(6, len(urls)))
    futures = {pool.submit(fetch, u, deadline): i for i, u in enumerate(urls)}
    try:
        for future in as_completed(futures, timeout=max(0.1, deadline - time.monotonic())):
            page = future.result()
            if page is not None and getattr(page, "text", "") and getattr(page, "status", 200) < 400:
                results[futures[future]] = page
    except Exception as err:  # a timeout here only means slow pages are dropped
        print(f"fetch: {err.__class__.__name__}", file=sys.stderr)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # never wait on a slow page past the deadline
    return [results[i] for i in sorted(results)]


def answer_web(query: str, reader: Reader, url: Optional[str] = None, corpus: Optional[Path] = None,
               sources: Optional[Sources] = None, start: Optional[float] = None) -> Tuple[str, List[str], float]:
    """One question in, (value, citations, confidence) out. Never raises."""
    start = time.monotonic() if start is None else start
    try:
        sources = sources or Sources()
        pages = gather_pages(query, url, corpus, sources, start + FETCH_BUDGET)
        chunks = [c for page in pages for c in sources.chunk_page(page)]
        if not chunks:
            return "", [], 0.0
        context = web_context(chunks, query)
        if not context or time.monotonic() > start + TOTAL_BUDGET:
            return "", [], 0.0
        reply = reader.read(query, context)
        answer = strip_honorific(str(reply.get("answer", "")))
        value, cited = check_answer(answer, list(reply.get("files", [])), context)
        return value, cited, (0.9 if value else 0.0)
    except Exception as err:  # a crash must still produce a valid, empty answer
        print(f"web answer failed: {err.__class__.__name__}: {err}", file=sys.stderr)
        return "", [], 0.0
