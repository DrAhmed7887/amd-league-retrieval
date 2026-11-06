from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from engine import web

PAGES = {
    "/": ('<html><head><title>Acme Home</title><meta name="description" content="Acme makes rockets">'
          '<style>.x{}</style></head><body><h1>Welcome</h1><script>var secret=1;</script>'
          '<a href="contact.html#top">Contact us</a> <a href="/about.html">About</a> '
          '<a href="/pricing.html">Pricing</a> <a href="/old">Old</a> <a href="/table.html">Data</a> '
          '<a href="/missing">Gone</a> <a href="/file.bin">Download</a> <a href="/login">Login</a> '
          '<a href="mailto:a@b.c">mail</a> <a href="https://other.example.org/x">Ext</a></body></html>'),
    "/contact.html": "<title>Contact</title><p>Support phone: +1 555 0100</p>",
    "/about.html": "<title>About</title><p>Founded in 1999.</p>",
    "/pricing.html": "<title>Pricing</title><p>Plan costs 10 dollars.</p>",
    "/login": "<title>Login</title><p>Sign in</p>",
    "/table.html": ("<title>Data</title><table><tr><th>Name</th><th>Qty</th></tr>"
                    "<tr><td>Bolt</td><td>5</td></tr></table>"),
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/old":
            self.send_response(302)
            self.send_header("Location", "/about.html")
            self.end_headers()
        elif self.path == "/file.bin":
            self._send(b"\x00\x01\x02", "application/octet-stream")
        elif self.path == "/slow":
            time.sleep(2)
            self._send(b"late", "text/plain")
        elif self.path in PAGES:
            self._send(PAGES[self.path].encode(), "text/html; charset=utf-8")
        else:
            self._send(b"nope", "text/html", 404)

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def site(monkeypatch):
    monkeypatch.setenv("WEB_ALLOW_LOCAL", "1")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def dl(seconds=10.0):
    return time.monotonic() + seconds


def test_extraction_and_links(site):
    p = web.fetch(site + "/", dl())
    assert p.title == "Acme Home" and p.description == "Acme makes rockets"
    assert "Welcome" in p.text and "secret" not in p.text and ".x{}" not in p.text
    urls = [u for u, _ in p.links]
    assert site + "/contact.html" in urls  # relative resolved, fragment dropped
    assert not any(u.startswith("mailto") for u in urls)
    assert ("%s/about.html" % site, "About") in p.links


def test_table_cells(site):
    p = web.fetch(site + "/table.html", dl())
    assert "Name | Qty" in p.text.splitlines()
    assert "Bolt | 5" in p.text.splitlines()


def test_redirect_final_url(site):
    p = web.fetch(site + "/old", dl())
    assert p.final_url == site + "/about.html" and "Founded in 1999" in p.text


def test_404_and_binary_skipped(site):
    p = web.fetch(site + "/missing", dl())
    assert p is not None and p.status == 404
    assert web.fetch(site + "/file.bin", dl()) is None


def test_private_and_bad_scheme_refused(site, monkeypatch):
    port = site.rsplit(":", 1)[1]
    monkeypatch.delenv("WEB_ALLOW_LOCAL")
    assert web.fetch("http://127.0.0.1:%s/" % port, dl()) is None
    assert web.fetch("http://10.0.0.1/", dl()) is None
    assert web.fetch("http://169.254.169.254/latest/meta-data", dl()) is None
    assert web.fetch("http://user:pw@example.com/", dl()) is None
    for u in ("file:///etc/passwd", "ftp://example.com/x"):
        assert web.fetch(u, dl()) is None


def test_deadline_respected(site):
    t = time.monotonic()
    assert web.fetch(site + "/slow", t + 0.5) is None
    assert time.monotonic() - t < 1.5
    assert web.fetch(site + "/", time.monotonic() - 1) is None


def test_crawl_prefers_contact(site):
    pages = web.crawl(site + "/", "What is the support phone number?", dl(), max_pages=2)
    assert [p.final_url for p in pages] == [site + "/", site + "/contact.html"]


def test_crawl_junk_and_site_filters(site):
    pages = web.crawl(site + "/", "company", dl(), max_pages=20)
    urls = {p.final_url for p in pages}
    assert site + "/login" not in urls and not any("other.example.org" in u for u in urls)
    assert len(urls) == len(pages)  # de-duplicated (redirect target not fetched twice)


def test_crawl_max_pages_and_deadline(site):
    assert len(web.crawl(site + "/", "x", dl(), max_pages=3)) == 3
    assert web.crawl(site + "/", "x", time.monotonic() + 1.0) == []


def test_chunk_shape_and_size():
    text = "\n".join("Line %d %s" % (i, "word " * 20) for i in range(60))
    page = web.Page(url="u", final_url="http://x/y", text=text, title="My Title")
    chunks = web.chunk_page(page)
    assert len(chunks) > 1
    for i, c in enumerate(chunks, 1):
        assert set(c) == {"file", "location", "text"}
        assert c["file"] == "http://x/y" and c["location"] == "part %d" % i
        assert len(c["text"]) <= 1200 and c["text"].startswith("My Title\n")
    assert web.chunk_page(web.Page(url="u", text="")) == []
    long = web.chunk_page(web.Page(url="u", text="a" * 5000))
    assert all(len(c["text"]) <= 1200 for c in long)


def test_load_saved_pages(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.html").write_text("<title>A</title><p>Hello</p>")
    (tmp_path / "sub" / "b.txt").write_text("plain  text")
    (tmp_path / "skip.png").write_bytes(b"x")
    pages = {p.url: p for p in web.load_saved_pages(tmp_path)}
    assert set(pages) == {"a.html", "sub/b.txt"}
    assert pages["a.html"].title == "A" and pages["sub/b.txt"].text == "plain text"
