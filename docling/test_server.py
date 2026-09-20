"""Tests for the worker's HTTP surface, with the extractor stubbed.

server.py had no tests at all, which mattered because it is the only thing
that runs in production: every request-validation branch below (size, name,
suffix, busy) decides whether a document is read or refused, and each one maps
to a status code apps/web/server/docling.cjs branches on. Docling itself is not
needed for any of it.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import extract as extractor
import pytest
import server


@pytest.fixture()
def worker(monkeypatch):
    """A real socket server, so the HTTP semantics are genuinely exercised."""
    monkeypatch.setattr(extractor, "extract", lambda path, name: {
        "pages": [{"number": 1, "text": "ok", "status": "native",
                   "method": "docling", "truncated": False}],
        "total": 1, "truncatedPages": False})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def post(base, body, name="doc.pdf", headers=None):
    request = urllib.request.Request(f"{base}/extract", data=body, method="POST")
    request.add_header("Content-Type", "application/octet-stream")
    if name is not None:
        request.add_header("X-Document-Name", name)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_health_lists_the_formats_the_worker_accepts(worker):
    with urllib.request.urlopen(f"{worker}/health", timeout=10) as response:
        body = json.loads(response.read())
    assert response.status == 200
    assert body["service"] == "docling"
    # The Node client keeps its own copy of this list; if they drift, uploads
    # are sent 25 MB across the wire only to be refused.
    assert set(body["formats"]) == set(extractor.SUPPORTED)


def test_a_document_is_extracted(worker):
    status, body = post(worker, b"%PDF-1.4 fake")
    assert status == 200
    assert body["pages"][0]["text"] == "ok"


def test_an_empty_body_is_refused_rather_than_converted(worker):
    status, body = post(worker, b"")
    assert status == 400
    assert "Invalid document" in body["error"]


def test_a_body_over_the_limit_is_refused_without_reading_it(worker):
    """Declared-oversize is rejected on the header, before any bytes are read.

    Sent as a raw socket rather than through urllib because the worker answers
    and closes without draining the body, so a client that is still uploading
    sees a TCP reset instead of the 400. That is fine here (the web tier caps
    document uploads at 25 MB, so this branch is defence in depth) but it is
    why this cannot be written with urlopen.
    """
    import socket
    host, port = worker.rsplit(":", 1)
    connection = socket.create_connection(("127.0.0.1", int(port)), timeout=10)
    try:
        connection.sendall(
            b"POST /extract HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"X-Document-Name: doc.pdf\r\n"
            + f"Content-Length: {server.LIMIT + 1}\r\n".encode()
            + b"\r\n")
        response = connection.recv(200).decode("latin-1", "replace")
    finally:
        connection.close()
    assert "400" in response.split("\r\n")[0], response.splitlines()[:1]


def test_a_name_that_is_a_path_is_refused(worker):
    # The name picks the parser, so it must never be usable as a path.
    for name in ["../../etc/passwd.pdf", "a/b.pdf", "a\\b.pdf"]:
        status, _ = post(worker, b"data", name=name)
        assert status == 400, name


def test_a_missing_or_overlong_name_is_refused(worker):
    assert post(worker, b"data", name=None)[0] == 400
    assert post(worker, b"data", name="x" * 201 + ".pdf")[0] == 400


def test_an_unsupported_suffix_is_415_not_a_silent_empty_result(worker):
    # The whole point of the change: refuse by name rather than store nothing.
    status, body = post(worker, b"PK\x03\x04", name="archive.zip")
    assert status == 415
    assert ".zip" in body["error"]


def test_an_extractor_failure_is_422_and_quotes_no_document_content(worker, monkeypatch):
    secret = "PATIENT NAME: REDACTED-CANARY"

    def boom(path, name):
        raise RuntimeError(secret)

    monkeypatch.setattr(extractor, "extract", boom)
    status, body = post(worker, b"data")
    assert status == 422
    # Exception text can quote document content; it must not escape.
    assert secret not in json.dumps(body)


def test_a_second_request_while_busy_is_refused_with_503(worker):
    started, release = threading.Event(), threading.Event()

    def slow(path, name):
        started.set()
        release.wait(5)
        return {"pages": [], "total": 0, "truncatedPages": False}

    import unittest.mock
    with unittest.mock.patch.object(extractor, "extract", slow):
        first = threading.Thread(target=lambda: post(worker, b"data"))
        first.start()
        assert started.wait(5), "the first request never reached the extractor"
        status, body = post(worker, b"data")
        release.set()
        first.join(10)
    # One job at a time is deliberate: two concurrent conversions double peak
    # memory, and peak memory is what got a 334-page PDF OOM-killed.
    assert status == 503
    assert "busy" in body["error"].lower()


def test_an_unknown_route_is_404(worker):
    assert post(worker, b"data", headers={})[0] == 200
    request = urllib.request.Request(f"{worker}/nope", data=b"x", method="POST")
    try:
        urllib.request.urlopen(request, timeout=10)
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as err:
        assert err.code == 404
