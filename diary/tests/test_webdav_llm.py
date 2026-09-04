"""WebDAV client tests against a local HTTP server (no external services), plus
LLM marker-parsing unit tests."""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.llm import LLMClient
from agent.webdav import WebDAVClient


class FakeDAVHandler(BaseHTTPRequestHandler):
    """Minimal WebDAV-ish server: GET/PUT with ETag + If-Match on /dav/file.md."""

    server_version = "FakeDAV/1.0"
    state = {"body": b"hello\n", "version": 1}

    def log_message(self, *args):  # silence
        pass

    def _respond(self, code, body=b"", headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("missing.md"):
            self._respond(404)
            return
        self._respond(
            200,
            FakeDAVHandler.state["body"],
            {"ETag": f'"v{FakeDAVHandler.state["version"]}"'},
        )

    def do_PUT(self):
        if_match = self.headers.get("If-Match")
        current_etag = f'"v{FakeDAVHandler.state["version"]}"'
        if if_match is not None and if_match != current_etag:
            self._respond(412)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        FakeDAVHandler.state["version"] += 1
        FakeDAVHandler.state["body"] = body
        self._respond(204, b"", {"ETag": f'"v{FakeDAVHandler.state["version"]}"'})


@pytest.fixture
def dav_url():
    server = HTTPServer(("127.0.0.1", 0), FakeDAVHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/dav/"
    server.shutdown()


def test_get_returns_etag(dav_url):
    dav = WebDAVClient(dav_url, "u", "p")
    text, etag = dav.get_text("file.md")
    assert text == "hello\n"
    assert etag == '"v1"'


def test_put_with_stale_etag_conflicts_then_fresh_succeeds(dav_url):
    dav = WebDAVClient(dav_url, "u", "p")
    text, etag = dav.get_text("file.md")
    # another writer bumps the etag behind our backs
    FakeDAVHandler.state["version"] += 100
    ok, new_etag, status = dav.put("file.md", b"stale write", if_match=etag, max_retries=1)
    assert not ok and status == 412
    # re-GET (as CorpusStore._guarded_write does) then write
    text, etag = dav.get_text("file.md")
    ok, new_etag, status = dav.put("file.md", b"fresh write", if_match=etag)
    assert ok and status == 204
    final, _ = dav.get_text("file.md")
    assert final == "fresh write"


def test_get_missing_file(dav_url):
    dav = WebDAVClient(dav_url, "u", "p")
    text, etag = dav.get_text("missing.md")
    assert text is None and etag is None


def test_list_dir_uses_generic_webdav_base_path(monkeypatch):
    dav = WebDAVClient("https://cloud.example/dav/user/", "u", "p")

    class Response:
        status_code = 207
        text = """<?xml version="1.0"?>
        <d:multistatus xmlns:d="DAV:">
          <d:response><d:href>/dav/user/Notes/</d:href><d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
          <d:response><d:href>/dav/user/Notes/2026-09.md</d:href><d:propstat><d:prop><d:getetag>\"abc\"</d:getetag><d:getlastmodified>now</d:getlastmodified><d:resourcetype/></d:prop></d:propstat></d:response>
        </d:multistatus>"""

        def raise_for_status(self):
            return None

    monkeypatch.setattr(dav._client, "request", lambda *args, **kwargs: Response())
    assert dav.list_dir("Notes") == [{
        "name": "2026-09.md",
        "path": "Notes/2026-09.md",
        "etag": '"abc"',
        "lastmod": "now",
        "is_dir": False,
    }]
    dav.close()


# ---------------- LLM marker parsing ----------------


def test_strip_log_marker_ok():
    reply = "Some supportive prose.\n\n[LOG: ok]"
    visible, decision = LLMClient.strip_log_marker(reply)
    assert visible == "Some supportive prose."
    assert decision == "ok"


def test_strip_log_marker_skip():
    visible, decision = LLMClient.strip_log_marker("Formatting help.\n[LOG: skip]")
    assert visible == "Formatting help."
    assert decision == "skip"


def test_marker_mid_text_is_not_treated_as_marker():
    reply = "I would never write [LOG: skip] in the middle. That's it."
    visible, decision = LLMClient.strip_log_marker(reply)
    assert decision is None
    assert "[LOG: skip]" in visible


def test_missing_marker_defaults_to_none():
    visible, decision = LLMClient.strip_log_marker("Just prose.")
    assert decision is None and visible == "Just prose."
