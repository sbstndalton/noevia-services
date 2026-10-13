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

from agent.llm import LLMClient, LLMError
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


def test_llm_client_still_follows_redirects_for_chat_but_never_for_embeddings():
    """The chat endpoint is operator-configured (config.yaml), not user-supplied. The embeddings
    server can be pointed elsewhere at runtime (#1636), so embedding calls refuse redirects on both
    paths: a dedicated server and the chat engine."""
    client = LLMClient(base_url="http://example.invalid", chat_model="m", embed_model="e", timeout_s=5)
    dedicated = LLMClient(base_url="http://example.invalid", chat_model="m", embed_model="e", timeout_s=5,
                          embed_base_url="http://embed.invalid:8080/v1")
    try:
        assert client._client.follow_redirects is True
        assert dedicated._client.follow_redirects is True
        assert client._embed_client.follow_redirects is False
        assert client._embed_client is not client._client
        assert dedicated._embed_client.follow_redirects is False
        # The engine key still goes to the engine's embeddings path, and never to a dedicated server.
        keyed = LLMClient(base_url="http://example.invalid", api_key="engine-key", embed_base_url="http://embed.invalid:8080/v1", timeout_s=5)
        try:
            assert keyed._embed_client.headers.get("Authorization") is None
        finally:
            keyed.close()
        shared = LLMClient(base_url="http://example.invalid", api_key="engine-key", timeout_s=5)
        try:
            assert shared._embed_client.headers["Authorization"] == "Bearer engine-key"
        finally:
            shared.close()
    finally:
        client.close()
        dedicated.close()


class _Recorder(BaseHTTPRequestHandler):
    """Answers every POST with 307 to `server.location` (when set) and records what it was sent."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.hits.append((self.path, body))
        if getattr(self.server, "location", None):
            self.send_response(307)  # keeps the method and the body: the worst case
            self.send_header("Location", self.server.location)
            self.send_header("Content-Length", "0")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            payload = b'{"data": [{"index": 0, "embedding": [0.1, 0.2]}]}'
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.end_headers()


@pytest.fixture()
def bounce():
    """A server that redirects POSTs to a second server, and the second server."""
    servers = []

    def start(location=None):
        srv = HTTPServer(("127.0.0.1", 0), _Recorder)
        srv.hits = []
        srv.location = location
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv

    sink = start()
    redirector = start(f"http://127.0.0.1:{sink.server_address[1]}/v1/embeddings")
    yield redirector, sink
    for srv in servers:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("dedicated", [True, False], ids=["dedicated embeddings server", "chat engine"])
def test_embeddings_refuse_redirects_and_send_the_text_once(bounce, dedicated):
    redirector, sink = bounce
    origin = f"http://127.0.0.1:{redirector.server_address[1]}/v1"
    client = LLMClient(base_url="http://127.0.0.1:9/v1" if dedicated else origin, chat_model="m", embed_model="e", timeout_s=5,
                       max_retries=3, retry_backoff_s=0.0, embed_base_url=origin if dedicated else "")
    try:
        with pytest.raises(LLMError, match="redirected"):
            client.embed(["synthetic diary entry"])
    finally:
        client.close()
    assert len(redirector.hits) == 1, "refused at once, not retried (each retry would send the text again)"
    assert sink.hits == [], "nothing reached the address the redirect named"


def test_embeddings_still_work_without_a_redirect(bounce):
    _redirector, sink = bounce
    client = LLMClient(base_url="http://127.0.0.1:9/v1", chat_model="m", embed_model="e", timeout_s=5,
                       embed_base_url=f"http://127.0.0.1:{sink.server_address[1]}/v1")
    try:
        assert client.embed(["x"]) == [[0.1, 0.2]]
    finally:
        client.close()


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
