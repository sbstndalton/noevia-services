"""Redirect-hardening tests for the storage/LLM HTTP paths.

A server the user configures must not be able to bounce a request inward
(to RFC1918 or a cloud metadata address) and have the client follow it.
These tests pin that the corpus storage backends refuse redirects outright,
while the operator-configured LLM client may still follow them.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from agent.llm import LLMClient
from agent.s3_storage import S3CorpusBackend
from agent.util import ensure_not_redirect, make_client
from agent.webdav import WebDAVCorpusBackend

# Loopback port 9 (discard): a target that fails fast if anything ever
# follows the redirect instead of refusing it.
INNER_URL = "http://127.0.0.1:9/inner"


class RedirectHandler(BaseHTTPRequestHandler):
    """302s every request — simulates a server bouncing traffic inward."""

    server_version = "Redirector/1.0"

    def log_message(self, *args):  # silence
        pass

    def _redirect(self):
        self.send_response(302)
        self.send_header("Location", INNER_URL)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._redirect()

    def do_HEAD(self):
        self._redirect()

    def do_PUT(self):
        self._redirect()

    def do_PROPFIND(self):
        self._redirect()


@pytest.fixture()
def redirect_server():
    server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _base(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_webdav_corpus_backend_refuses_redirects(redirect_server):
    client = WebDAVCorpusBackend(base_url=_base(redirect_server), username="u", password="p", timeout_s=5)
    with pytest.raises(RuntimeError, match="redirected"):
        client.get_text("2026/2026-09.md")


def test_s3_corpus_backend_refuses_redirects(redirect_server):
    client = S3CorpusBackend(endpoint_url=_base(redirect_server), bucket="b", timeout_s=5)
    with pytest.raises(RuntimeError, match="redirected"):
        client.get("2026/2026-09.md")


def test_llm_client_still_follows_redirects():
    """The LLM endpoint is operator-configured (config.yaml), not user-supplied."""
    client = LLMClient(base_url="http://example.invalid", chat_model="m", embed_model="e", timeout_s=5)
    assert client._client.follow_redirects is True


def test_make_client_default_is_strict():
    client = make_client(base_url="http://example.invalid")
    try:
        assert client.follow_redirects is False
    finally:
        client.close()


def test_ensure_not_redirect_passes_real_statuses():
    response = httpx.Response(200)
    ensure_not_redirect(response)  # must not raise
    with pytest.raises(RuntimeError, match="redirected"):
        ensure_not_redirect(httpx.Response(302))
    with pytest.raises(RuntimeError, match="redirected"):
        ensure_not_redirect(httpx.Response(307))
