"""get() must not buffer an unbounded response body in memory (synthetic server)."""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.webdav import WebDAVCorpusBackend


class OversizedHandler(BaseHTTPRequestHandler):
    """Serves a body larger than the backend's default read cap."""

    server_version = "Oversized/1.0"

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("ETag", '"abc"')
        self.end_headers()
        chunk = b"x" * 65536
        try:
            for _ in range(2000):  # far past a small test cap
                self.wfile.write(chunk)
        except BrokenPipeError:
            pass


@pytest.fixture()
def oversized_server():
    server = HTTPServer(("127.0.0.1", 0), OversizedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_get_enforces_the_default_read_limit(oversized_server, monkeypatch):
    monkeypatch.setattr(WebDAVCorpusBackend, "DEFAULT_READ_LIMIT", 1024 * 1024)
    base = f"http://127.0.0.1:{oversized_server.server_address[1]}"
    client = WebDAVCorpusBackend(base_url=base, username="u", password="p", timeout_s=5)
    with pytest.raises(ValueError, match="safety limit"):
        client.get("big.md")
