"""Tests for the worker's HTTP surface, with the extractor stubbed.

server.py had no tests at all, which mattered because it is the only thing
that runs in production: every request-validation branch below (size, name,
suffix, busy) decides whether a document is read or refused, and each one maps
to a status code apps/web/server/docling.cjs branches on. Docling itself is not
needed for any of it.
"""
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import extract as extractor
import isolation
import pytest
import server


@pytest.fixture()
def worker(monkeypatch):
    """A real socket server, so the HTTP semantics are genuinely exercised.

    The conversion target is fake_converter.py, run in the REAL child process
    (isolation.py), so deadlines, kills and disconnects are the production code
    paths; only Docling is absent. The body of each document is a directive.
    """
    runner = isolation.IsolatedExtractor("fake_converter:extract")
    monkeypatch.setattr(server, "runner", runner)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        runner.kill()
        server.runner.kill()  # a test may have swapped in its own


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_until(predicate, seconds=10):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


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


def test_an_extractor_failure_is_422_and_quotes_no_document_content(worker):
    import fake_converter
    status, body = post(worker, b"BOOM")
    assert status == 422
    # Exception text can quote document content; it must not escape.
    assert fake_converter.SECRET not in json.dumps(body)


def test_a_failed_conversion_keeps_the_warm_child_for_the_next_document(worker):
    # The converter holds 669 MB of models; an ordinary failure must not throw them away.
    first = post(worker, b"go")[1]["pid"]
    assert post(worker, b"BOOM")[0] == 422
    assert post(worker, b"go")[1]["pid"] == first


def test_a_second_request_while_busy_is_refused_with_503(worker):
    first = threading.Thread(target=lambda: post(worker, b"SLEEP 3"))
    first.start()
    assert wait_until(lambda: server.runner.pid is not None), "the first request never reached the extractor"
    status, body = post(worker, b"data")
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


# --- per-document limits (#854) ---------------------------------------------

def test_each_input_type_has_a_time_limit_and_pdfs_keep_the_long_one():
    assert server.timeout_for(".pdf") == server.LONG_TIMEOUT
    assert server.timeout_for(".tiff") == server.LONG_TIMEOUT
    for suffix in [".xlsx", ".ods", ".docx", ".pptx", ".odt", ".odp", ".html", ".htm", ".md", ".epub", ".csv", ".png", ".jpg"]:
        assert server.timeout_for(suffix) == server.SHORT_TIMEOUT, suffix
    # Under the web client's 65 min abort, so it gets a 422 rather than giving up itself.
    assert server.LONG_TIMEOUT < 3900
    assert server.SHORT_TIMEOUT < server.LONG_TIMEOUT
    # Every supported suffix resolves to one of the two limits.
    assert {server.timeout_for(s) for s in extractor.SUPPORTED} <= {server.LONG_TIMEOUT, server.SHORT_TIMEOUT}


@pytest.mark.parametrize("name", ["big.xlsx", "page.html", "deck.pptx", "notes.md", "scan.png"])
def test_a_non_pdf_that_never_finishes_is_cut_off_with_a_422_and_frees_the_slot(worker, monkeypatch, name):
    monkeypatch.setattr(server, "SHORT_TIMEOUT", 1.0)
    started = time.monotonic()
    status, body = post(worker, b"SLEEP 60", name=name)
    assert status == 422
    assert time.monotonic() - started < 8
    assert "time limit" in body["error"]
    # The slot is free again and the next document is read by a fresh child.
    status, body = post(worker, b"fine", name="next.xlsx")
    assert status == 200
    assert body["pages"][0]["text"] == "ok"


def test_a_pdf_is_held_to_the_long_limit_not_the_short_one(worker, monkeypatch):
    monkeypatch.setattr(server, "SHORT_TIMEOUT", 0.2)
    monkeypatch.setattr(server, "LONG_TIMEOUT", 1.0)
    # 0.6 s is past the short limit and inside the long one: a PDF finishes, an XLSX does not.
    assert post(worker, b"SLEEP 0.6", name="book.pdf")[0] == 200
    assert post(worker, b"SLEEP 0.6", name="sheet.xlsx")[0] == 422
    # And the long limit still ends a PDF that never finishes.
    assert post(worker, b"SLEEP 60", name="book.pdf")[0] == 422


def test_the_deadline_kills_the_conversion_process_and_what_it_started(worker, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SHORT_TIMEOUT", 1.5)
    pidfile = tmp_path / "grandchild.pid"
    done = {}
    thread = threading.Thread(target=lambda: done.update(r=post(worker, f"FORK {pidfile}".encode(), name="x.xlsx")))
    thread.start()
    assert wait_until(lambda: pidfile.exists() and pidfile.read_text())
    child, grandchild = server.runner.pid, int(pidfile.read_text())
    assert alive(child) and alive(grandchild)
    thread.join(15)
    assert done["r"][0] == 422
    assert wait_until(lambda: not alive(child)), "the conversion process outlived its deadline"
    assert wait_until(lambda: not alive(grandchild)), "a process the conversion started outlived it"
    assert server.runner.pid is None


def test_a_client_that_goes_away_stops_the_conversion_and_frees_the_slot(worker, tmp_path):
    host, port = worker.rsplit(":", 1)
    body = b"SLEEP 60"
    connection = socket.create_connection(("127.0.0.1", int(port)), timeout=10)
    connection.sendall(
        b"POST /extract HTTP/1.1\r\nHost: localhost\r\nX-Document-Name: big.xlsx\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    assert wait_until(lambda: server.runner.pid is not None), "the conversion never started"
    child = server.runner.pid
    connection.close()  # the web tier gave up
    assert wait_until(lambda: not alive(child)), "the conversion kept running for a client that left"
    assert wait_until(lambda: server.runner.pid is None)
    # Slot released: a new document is accepted rather than refused as busy.
    assert wait_until(lambda: post(worker, b"fine")[0] == 200)


def test_a_conversion_process_that_dies_is_a_422_and_the_next_document_still_works(worker):
    # An OOM kill (exit 137) inside the child must not take the worker down with it.
    status, body = post(worker, b"EXIT", name="big.xlsx")
    assert status == 422
    assert "could not be read" in body["error"]
    assert post(worker, b"fine")[0] == 200


def test_the_child_is_reused_so_models_load_once(worker):
    first = post(worker, b"one")[1]["pid"]
    assert post(worker, b"two", name="b.docx")[1]["pid"] == first
    assert first != os.getpid()


# --- the conversion process cannot start: infrastructure, so 503 and never a cached 422 ------------

def test_a_worker_that_cannot_import_its_target_is_503_not_a_permanent_422(worker, monkeypatch):
    monkeypatch.setattr(server, "runner", isolation.IsolatedExtractor("no_such_module:extract"))
    status, body = post(worker, b"data")
    assert status == 503
    assert "unavailable" in body["error"].lower()
    assert server.runner.pid is None


def test_a_worker_that_cannot_be_spawned_is_503(worker, monkeypatch):
    monkeypatch.setattr(isolation.sys, "executable", "/nonexistent/python")
    status, _ = post(worker, b"data")
    assert status == 503


def test_a_worker_that_never_becomes_ready_is_503_and_is_killed(worker, monkeypatch):
    monkeypatch.setattr(isolation, "READY_SECONDS", 1.0)
    monkeypatch.setattr(server, "runner", isolation.IsolatedExtractor("fake_slow_import:extract"))
    started = time.monotonic()
    status, _ = post(worker, b"data")
    assert status == 503
    assert time.monotonic() - started < 8
    assert server.runner.pid is None


def test_a_recovered_worker_serves_the_next_document(worker, monkeypatch):
    broken = isolation.IsolatedExtractor("no_such_module:extract")
    monkeypatch.setattr(server, "runner", broken)
    assert post(worker, b"data")[0] == 503
    broken.target = "fake_converter:extract"
    assert post(worker, b"data")[0] == 200


def test_a_large_result_arrives_whole(worker, monkeypatch):
    # Frames are flushed whole: a result bigger than a pipe buffer (64 KiB) must not be truncated.
    big = {"pages": [{"number": 1, "text": "x" * 190_000, "status": "native", "method": "docling", "truncated": False}],
           "total": 1, "truncatedPages": False}
    monkeypatch.setattr(server, "runner", isolation.IsolatedExtractor("fake_converter:big"))
    status, body = post(worker, b"data")
    assert status == 200
    assert body["pages"][0]["text"] == "x" * 190_000
