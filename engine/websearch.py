"""Find candidate page URLs for a question that comes without a start URL.

Returns URLs of pages (never answers). Stdlib only; every HTTP call goes
through an injectable ``opener(url, timeout) -> (status, body)``.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Callable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse, urlunparse

Opener = Callable[[str, float], Tuple[int, str]]

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
MAX_TIMEOUT = 10.0
MAX_BYTES = 1_500_000

_FILLER = {"what", "which", "who", "whom", "when", "where", "why", "how", "is",
           "are", "was", "were", "does", "did", "do", "the", "a", "an",
           "please", "tell", "me", "can", "you", "could", "give", "find"}
_NOT_NAMES = {"package", "version", "library", "module", "project", "latest",
              "current", "the", "a", "an", "of", "python", "node", "new",
              "newest", "release", "on", "in", "from"}
_TLDS = "com|org|net|io|dev|gov|edu|co|ai|app|info|int|eu|uk|de|es|fr|wiki"
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)
_DOMAIN_RE = re.compile(
    r"(?<![@\w./-])((?:[a-z0-9-]+\.)+(?:%s)(?:/[^\s<>\"']*)?)(?![\w@-])" % _TLDS, re.I)
_VERSION_RE = re.compile(
    r"\b(latest|current|newest|most recent)\b.*\b(version|release)\b"
    r"|\b(version|release)\b.*\b(latest|current|newest)\b"
    r"|\b(what|which) (version|release)\b", re.I)
_NAME = r"([A-Za-z0-9][\w.-]*)"


def is_version_question(query: str) -> bool:
    return bool(_VERSION_RE.search(query or ""))


def clean_query(query: str) -> str:
    """Search string: drop question words and filler, keep entities/numbers/quotes."""
    quoted = re.findall(r'"[^"]+"', query)
    rest = re.sub(r'"[^"]+"', " ", query)
    tokens = [t.strip("?!.,;:()") for t in rest.split()]
    kept = [t for t in tokens if t and t.lower() not in _FILLER]
    out = " ".join(kept + quoted).strip()
    return out or query.strip(" ?!.")


def normalize_url(url: str) -> str:
    p = urlparse(url.strip())
    path = p.path.rstrip("/")
    return urlunparse((p.scheme, p.netloc, path, p.params, p.query, ""))


def urls_in_text(text: str) -> List[str]:
    strip = ".,;:!?)]}"
    found = [m.group(0).rstrip(strip) for m in _URL_RE.finditer(text)]
    bare = _URL_RE.sub(" ", text)
    found += ["https://" + m.group(1).rstrip(strip) for m in _DOMAIN_RE.finditer(bare)]
    return found


def _fetch_ok(call: Opener, url: str) -> bool:
    status, _ = call(url, MAX_TIMEOUT)
    return status == 200


def _names(query: str, patterns: List[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, query, re.I):
            name = m.group(1).strip(".-")
            if name and name.lower() not in _NOT_NAMES and name not in out:
                out.append(name)
    return out


def registry_urls(query: str, call: Opener) -> List[str]:
    text = _URL_RE.sub(" ", query)
    out: List[str] = []
    pypi = _names(text, [
        _NAME + r"\s+(?:package\s+|library\s+|project\s+|module\s+)?(?:on|from|in|at)\s+pypi\b",
        r"\bpypi\s+(?:package\s+|project\s+|library\s+)?[\"']?" + _NAME,
        r"\bpip\s+install\s+" + _NAME])
    for name in pypi[:2]:
        if _fetch_ok(call, "https://pypi.org/pypi/%s/json" % quote(name)):
            out.append("https://pypi.org/project/%s/" % name.lower())
            break
    npm_name = r"(@?[A-Za-z0-9][\w.-]*(?:/[\w.-]+)?)"
    npm = _names(text, [
        r"\bnpm\s+(?:package\s+|module\s+|library\s+)" + npm_name,
        npm_name + r"\s+(?:package\s+)?(?:on|from|in)\s+npm\b",
        r"\bnpm\s+install\s+" + npm_name])
    for name in npm[:2]:
        if _fetch_ok(call, "https://registry.npmjs.org/%s" % quote(name, safe="@")):
            out.append("https://www.npmjs.com/package/%s" % name)
            break
    if re.search(r"github|release", text, re.I):
        for owner, repo in re.findall(r"\b([\w.-]+)/([\w.-]+)\b", text)[:2]:
            url = "https://github.com/%s/%s/releases/latest" % (owner, repo)
            if _fetch_ok(call, url):
                out.append(url)
                break
    return out


def wikipedia_urls(q: str, call: Opener) -> List[str]:
    url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s"
           "&format=json&srlimit=3" % quote(q))
    _, body = call(url, MAX_TIMEOUT)
    try:
        hits = json.loads(body)["query"]["search"]
        titles = [h["title"] for h in hits]
    except (ValueError, KeyError, TypeError):
        return []
    return ["https://en.wikipedia.org/wiki/" + quote(t.replace(" ", "_"), safe="_(),:!'")
            for t in titles if isinstance(t, str)]


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _unwrap(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    p = urlparse(href)
    host = (p.hostname or "")
    if host.endswith("duckduckgo.com") and p.path.startswith("/l/"):
        vals = parse_qs(p.query).get("uddg")
        return vals[0] if vals else ""
    if host.endswith("bing.com") and p.path.startswith("/ck/"):
        vals = parse_qs(p.query).get("u")
        if vals and vals[0].startswith("a1"):
            raw = vals[0][2:]
            try:
                return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return ""
    return href


def _usable(url: str, banned: str) -> bool:
    p = urlparse(url)
    return p.scheme in ("http", "https") and bool(p.hostname) \
        and not _host(url).endswith(banned)


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and "result__a" in (a.get("class") or "").split() and a.get("href"):
            self.hrefs.append(a["href"])


class _BingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []
        self._algo = self._h2 = self._got = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "li" and "b_algo" in (a.get("class") or "").split():
            self._algo, self._got = True, False
        elif tag == "h2" and self._algo:
            self._h2 = True
        elif tag == "a" and self._h2 and not self._got and a.get("href"):
            self.hrefs.append(a["href"])
            self._got = True

    def handle_endtag(self, tag):
        if tag == "h2":
            self._h2 = False
        elif tag == "li":
            self._algo = False


def parse_ddg(html: str) -> List[str]:
    p = _DDGParser()
    try:
        p.feed(html)
    except Exception:  # malformed markup: use what we got
        pass
    out = []
    for href in p.hrefs:
        if "y.js" in href or "ad_provider" in href:
            continue
        url = _unwrap(href)
        if url and _usable(url, "duckduckgo.com"):
            out.append(url)
    return out


def parse_bing(html: str) -> List[str]:
    p = _BingParser()
    try:
        p.feed(html)
    except Exception:
        pass
    urls = [_unwrap(h) for h in p.hrefs]
    return [u for u in urls if u and _usable(u, "bing.com")]


def web_urls(q: str, call: Opener) -> List[str]:
    _, body = call("https://html.duckduckgo.com/html/?q=" + quote(q), MAX_TIMEOUT)
    found = parse_ddg(body)
    if found:
        return found
    _, body = call("https://www.bing.com/search?q=" + quote(q), MAX_TIMEOUT)
    return parse_bing(body)


def default_opener(url: str, timeout: float) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(MAX_BYTES).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def _guarded(base: Opener, deadline: float) -> Opener:
    def call(url: str, timeout: float) -> Tuple[int, str]:
        left = deadline - time.monotonic()
        if left <= 0:
            return 0, ""
        try:
            status, body = base(url, max(0.1, min(timeout, left)))
            return int(status), body if isinstance(body, str) else ""
        except Exception:
            return 0, ""
    return call


def candidate_urls(query: str, deadline: float, k: int = 8,
                   opener: Optional[Opener] = None) -> List[str]:
    seen: List[str] = []

    def add(urls: List[str]) -> None:
        for u in urls:
            n = normalize_url(u)
            if n and n not in seen and len(seen) < k:
                seen.append(n)

    add(urls_in_text(query))
    if deadline - time.monotonic() <= 0 or len(seen) >= k:
        return seen
    call = _guarded(opener or default_opener, deadline)
    q = clean_query(query)
    pool = ThreadPoolExecutor(max_workers=3)
    futs = [pool.submit(registry_urls, query, call),
            pool.submit(web_urls, q, call),
            pool.submit(wikipedia_urls, q, call)]
    results: List[List[str]] = []
    for f in futs:
        try:
            results.append(f.result(timeout=max(0.0, deadline - time.monotonic())))
        except Exception:
            results.append([])
    pool.shutdown(wait=False, cancel_futures=True)
    registry, web, wiki = results
    add(registry)
    # Interleave the search engine with Wikipedia, web first: the engine's top hit is usually the
    # official page, while Wikipedia full-text search is noisy for long questions.
    for pair in zip(web + [""] * len(wiki), wiki + [""] * len(web)):
        add([u for u in pair if u])
    return seen
