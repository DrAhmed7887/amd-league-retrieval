"""Web fetching, HTML-to-text, chunking and a small query-guided crawler (stdlib only)."""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlsplit

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
OK_TYPES = ("text/html", "text/plain", "application/json", "application/xhtml+xml")
SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "head"}
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "section", "article", "header", "footer",
              "nav", "main", "table", "pre", "blockquote", "dt", "dd", "form", "tr"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
JUNK = re.compile(r"login|log-in|signin|sign-in|signup|sign-up|register|cart|checkout|privacy|"
                  r"terms|cookie|unsubscribe|logout|"
                  # Wikipedia/MediaWiki pages that are never the article itself
                  r"/wiki/(Talk|Special|User|User_talk|Wikipedia|Help|File|Category|Portal|Template|Draft):|"
                  r"[?&](action|oldid|diff|printable)=", re.I)
JUNK_EXT = re.compile(r"\.(pdf|zip|gz|jpe?g|png|gif|svg|webp|mp[34]|avi|exe|dmg|css|js|ico|docx?|xlsx?)$", re.I)
STOP = {"the", "what", "is", "are", "of", "a", "an", "and", "for", "to", "in", "on", "does",
        "do", "how", "who", "which", "when", "where", "their", "its", "this", "that", "with"}
SYNONYMS = {"phone": "contact", "call": "contact", "email": "contact", "mail": "contact",
            "address": "contact", "support": "contact", "reach": "contact",
            "price": "pricing", "cost": "pricing", "fee": "pricing", "plan": "pricing",
            "founded": "about", "founder": "about", "history": "about", "team": "about"}


@dataclass
class Page:
    url: str
    final_url: str = ""
    status: int = 200
    text: str = ""
    title: str = ""
    links: List[Tuple[str, str]] = field(default_factory=list)
    description: str = ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # surface 3xx as HTTPError; we follow manually
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _host_allowed(host: str) -> bool:
    allow_local = os.environ.get("WEB_ALLOW_LOCAL") == "1"
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_loopback and allow_local:
            continue
        # Only public internet addresses: is_global also rules out CGNAT 100.64/10 and NAT64 ranges.
        if not ip.is_global or ip.is_multicast:
            return False
    return bool(infos)


def _url_ok(url: str) -> bool:
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
        return False
    return _host_allowed(p.hostname)


def _decode(raw: bytes, ctype: str) -> str:
    m = re.search(r"charset=([\w-]+)", ctype, re.I)
    if not m:
        m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", raw[:2048], re.I)
        enc = m.group(1).decode("ascii", "ignore") if m else "utf-8"
    else:
        enc = m.group(1)
    try:
        return raw.decode(enc, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _request(url: str, deadline: float, timeout: float, max_bytes: int):
    left = deadline - time.monotonic()
    if left <= 0.05:
        return None
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1"})
    try:
        resp = _OPENER.open(req, timeout=min(timeout, left))
    except urllib.error.HTTPError as e:
        return e
    return resp


def _read_until(resp, deadline: float, max_bytes: int) -> bytes:
    """Read in pieces so a slow-drip server cannot hold us past the deadline."""
    parts, size = [], 0
    while size < max_bytes and time.monotonic() < deadline:
        piece = resp.read(min(65536, max_bytes - size))
        if not piece:
            break
        parts.append(piece)
        size += len(piece)
    return b"".join(parts)


def fetch(url: str, deadline: float, max_bytes: int = 2_000_000, timeout: float = 8.0) -> Optional[Page]:
    """Fetch one page safely. Returns None on any failure or unsupported content."""
    try:
        cur = url
        for _ in range(6):
            if not _url_ok(cur):
                return None
            resp = _request(cur, deadline, timeout, max_bytes)
            if resp is None:
                return None
            code = resp.getcode()
            if code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location")
                if not loc:
                    return None
                cur = urldefrag(urljoin(cur, loc))[0]
                continue
            ctype = resp.headers.get("Content-Type", "text/html")
            if not any(ctype.lower().startswith(t) for t in OK_TYPES):
                return None
            raw = _read_until(resp, deadline, max_bytes)
            body = _decode(raw, ctype)
            page = Page(url=url, final_url=cur, status=code)
            if "html" in ctype.lower():
                _fill_from_html(page, body)
            else:
                page.text = _clean_lines(body)
            return page
        return None
    except Exception as e:  # never raise
        print("web.fetch failed: %s: %s" % (url, e), file=sys.stderr)
        return None


class _Extractor(HTMLParser):
    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base, self.out, self.links = base, [], []
        self.title, self.description = "", ""
        self._skip, self._in_title, self._href, self._anchor = 0, False, None, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta" and (a.get("name") or "").lower() == "description":
            self.description = a.get("content") or ""
        if tag in SKIP_TAGS - {"head"}:
            self._skip += 1
        if self._skip:
            return
        if tag in BLOCK_TAGS or tag in HEADINGS:
            self.out.append("\n")
        if tag == "a" and a.get("href"):
            self._href, self._anchor = a["href"].strip(), []

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in SKIP_TAGS - {"head"}:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in BLOCK_TAGS or tag in HEADINGS:
            self.out.append("\n")
        if tag == "a" and self._href is not None:
            self._add_link()

    def _add_link(self):
        href, self._href = self._href, None
        if re.match(r"(mailto|javascript|tel|data):", href, re.I) or href.startswith("#"):
            return
        url = urldefrag(urljoin(self.base, href))[0]
        if url.startswith(("http://", "https://")):
            self.links.append((url, " ".join("".join(self._anchor).split())))

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._skip:
            return
        if not self._in_title:
            self.out.append(data)
        if self._href is not None:
            self._anchor.append(data)


def _clean_lines(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        ln = " ".join(ln.split()).strip()
        ln = re.sub(r"(\s*\|)+$", "", ln).strip()
        ln = re.sub(r"^(\|\s*)+", "", ln)
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def _fill_from_html(page: Page, html: str) -> None:
    p = _Extractor(page.final_url)
    try:
        p.feed(html)
        p.close()
    except Exception as e:
        print("web: html parse problem: %s" % e, file=sys.stderr)
    page.title = " ".join(p.title.split())
    page.description = " ".join(p.description.split())
    page.text = _clean_lines("".join(p.out))
    page.links = p.links


def _split_long(line: str, size: int, overlap: int) -> List[str]:
    step = max(1, size - overlap)
    return [line[i:i + size] for i in range(0, len(line), step)]


def chunk_page(page: Page, size: int = 1200, overlap: int = 150) -> List[dict]:
    """Split page text into {"file","location","text"} chunks, breaking at line boundaries."""
    if not page.text.strip():
        return []
    prefix = (page.title + "\n") if page.title else ""
    budget = max(200, size - len(prefix))
    lines: List[str] = []
    for ln in page.text.split("\n"):
        lines.extend(_split_long(ln, budget, overlap) if len(ln) > budget else [ln])
    groups, cur, cur_len = [], [], 0
    for ln in lines:
        if cur and cur_len + len(ln) + 1 > budget:
            groups.append(cur)
            tail, n = [], 0
            for prev in reversed(cur):  # carry a short overlap of trailing lines
                if n + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                n += len(prev) + 1
            cur, cur_len = tail, n
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        groups.append(cur)
    return [{"file": page.final_url or page.url, "location": "part %d" % i,
             "text": prefix + "\n".join(g)} for i, g in enumerate(groups, 1)]


def _norm(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")


def _site(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _tokens(s: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) >= 3 and t not in STOP}


def _link_score(query_tokens: Set[str], url: str, anchor: str) -> int:
    words = _tokens(anchor) | _tokens(urlsplit(url).path)
    expanded = query_tokens | {SYNONYMS[t] for t in query_tokens if t in SYNONYMS}
    return len(expanded & words)


def _is_junk(url: str, anchor: str, query: str) -> bool:
    path = urlsplit(url).path
    q = query.lower()
    if JUNK.search(path + " " + anchor) and not JUNK.search(q):
        return True
    m = JUNK_EXT.search(path)
    return bool(m and m.group(1).lower() not in q)


def crawl(start_url: str, query: str, deadline: float, max_pages: int = 12,
          max_depth: int = 2, same_site: bool = True) -> List[Page]:
    """Best-first crawl from start_url, following links that overlap with the query."""
    qt = _tokens(query)
    seen = {_norm(start_url)}
    frontier: List[Tuple[int, int, int, str]] = [(0, 0, 0, start_url)]  # (-score, order, depth, url)
    pages: List[Page] = []
    order = 1
    while frontier and len(pages) < max_pages and deadline - time.monotonic() >= 1.5:
        frontier.sort()
        _, _, depth, url = frontier.pop(0)
        page = fetch(url, deadline)
        if page is None or page.status >= 400 or _norm(page.final_url) in {_norm(x.final_url) for x in pages}:
            continue
        pages.append(page)
        seen.add(_norm(page.final_url))
        if depth >= max_depth:
            continue
        for link, anchor in page.links:
            key = _norm(link)
            if key in seen or _is_junk(link, anchor, query):
                continue
            if same_site and _site(link) != _site(start_url):
                continue
            seen.add(key)
            frontier.append((-_link_score(qt, link, anchor), order, depth + 1, link))
            order += 1
    return pages


def load_saved_pages(directory) -> List[Page]:
    """Offline mode: read saved .html/.htm/.txt files; url = posix path relative to directory."""
    root = Path(directory)
    pages: List[Page] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".html", ".htm", ".txt"):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            body = path.read_bytes().decode("utf-8", errors="replace")
        except OSError as e:
            print("web: cannot read %s: %s" % (path, e), file=sys.stderr)
            continue
        page = Page(url=rel, final_url=rel, status=200)
        if path.suffix.lower() == ".txt":
            page.text = _clean_lines(body)
        else:
            _fill_from_html(page, body)
        pages.append(page)
    return pages
