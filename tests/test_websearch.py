from __future__ import annotations

import base64
import json
import time

from engine import websearch as ws

DDG = """
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%2F%23frag&rut=x">A</a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad_provider=bing&u=1">Ad</a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/about">DDG</a></div>
<div class="result"><a class="result__a" href="https://direct.example.com/page">B</a></div>
<a class="other" href="https://nope.example.com/">no</a>
"""
BING = """
<ol><li class="b_algo"><h2><a href="https://bing-hit.example.com/x">T</a></h2><a href="https://skip.example.com">s</a></li>
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u=a1%s&x=1">T2</a></h2></li></ol>
""" % base64.urlsafe_b64encode(b"https://wrapped.example.net/p").decode().rstrip("=")


def wiki_json(*titles):
    return json.dumps({"query": {"search": [{"title": t} for t in titles]}})


class Fake:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return 404, ""


def run(q, routes, k=8, left=5.0):
    f = Fake(routes)
    return ws.candidate_urls(q, time.monotonic() + left, k=k, opener=f), f


def test_url_in_question_first():
    urls, _ = run("What does https://docs.python.org/3/faq/ say about GIL?",
                  {"https://en.wikipedia.org": (200, wiki_json("Global interpreter lock"))})
    assert urls[0] == "https://docs.python.org/3/faq"
    assert "https://en.wikipedia.org/wiki/Global_interpreter_lock" in urls


def test_bare_domain():
    assert ws.urls_in_text("Who founded python.org?") == ["https://python.org"]
    assert ws.urls_in_text("mail me@x.com about file.py") == []


def test_pypi_hit():
    urls, f = run("What is the latest version of the requests package on PyPI?",
                  {"https://pypi.org/pypi/requests/json": (200, "{}")})
    assert "https://pypi.org/project/requests" in urls
    assert ws.is_version_question("What is the latest version of requests?")
    assert not ws.is_version_question("Who founded Python?")


def test_pypi_miss():
    urls, _ = run("latest version of nopkg on PyPI", {"https://pypi.org/pypi/nopkg/json": (404, "")})
    assert not any("pypi.org/project" in u for u in urls)


def test_npm():
    urls, _ = run("npm package react latest version",
                  {"https://registry.npmjs.org/react": (200, "{}")})
    assert "https://www.npmjs.com/package/react" in urls


def test_github_release():
    urls, _ = run("latest release of github psf/requests",
                  {"https://github.com/psf/requests/releases/latest": (200, "x")})
    assert "https://github.com/psf/requests/releases/latest" in urls


def test_wikipedia_parse_and_bad_json():
    f = Fake({"https://en.wikipedia.org": (200, wiki_json("Foo Bar (band)", "Baz"))})
    assert ws.wikipedia_urls("foo", f) == [
        "https://en.wikipedia.org/wiki/Foo_Bar_(band)", "https://en.wikipedia.org/wiki/Baz"]
    assert ws.wikipedia_urls("foo", Fake({"https://en.wikipedia.org": (200, "<html>")})) == []


def test_ddg_parse_unwrap_and_ads():
    assert ws.parse_ddg(DDG) == ["https://example.org/a/#frag", "https://direct.example.com/page"]
    urls, _ = run("anything", {"https://html.duckduckgo.com": (200, DDG)})
    assert urls == ["https://example.org/a", "https://direct.example.com/page"]


def test_bing_only_when_ddg_empty():
    urls, f = run("q", {"https://html.duckduckgo.com": (200, DDG), "https://www.bing.com": (200, BING)})
    assert not any("bing.com" in c for c in f.calls)
    urls, f = run("q", {"https://html.duckduckgo.com": (200, "<html></html>"),
                        "https://www.bing.com": (200, BING)})
    assert urls == ["https://bing-hit.example.com/x", "https://wrapped.example.net/p"]


def test_dedup_and_k():
    ddg = "".join('<a class="result__a" href="https://e.com/%d/">x</a>' % (i % 3) for i in range(10))
    urls, _ = run("q", {"https://html.duckduckgo.com": (200, ddg)})
    assert urls == ["https://e.com/0", "https://e.com/1", "https://e.com/2"]
    many = "".join('<a class="result__a" href="https://e.com/%d">x</a>' % i for i in range(20))
    urls, _ = run("q", {"https://html.duckduckgo.com": (200, many)}, k=4)
    assert len(urls) == 4


def test_deadline_passed_no_requests():
    f = Fake({})
    urls = ws.candidate_urls("see python.org and https://a.org/x#y", time.monotonic() - 1, opener=f)
    assert urls == ["https://a.org/x", "https://python.org"]
    assert f.calls == []


def test_failures_tolerated():
    urls, _ = run("who founded foo", {"https://en.wikipedia.org": RuntimeError("boom"),
                                      "https://html.duckduckgo.com": (0, ""),
                                      "https://www.bing.com": (0, "")})
    assert urls == []


def test_clean_query():
    assert ws.clean_query("What is the population of Valencia in 2020?") == "population of Valencia in 2020"
    assert '"Black Swan"' in ws.clean_query('Please tell me who directed "Black Swan"?')
