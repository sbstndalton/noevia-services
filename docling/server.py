"""Private, stateless Docling extraction worker.

Same posture as services/ocr: no persistent volume, no corpus credentials,
no logging of document contents or names, one job at a time, generic errors.

It is a SEPARATE service from services/ocr on purpose. The OCR worker is a
thin shell around poppler and tesseract; this one holds ~1.6 GB of layout and
table models resident. Keeping them apart means the operator can run, restart,
scale or disable either without touching the other, and a Docling model load
never delays a plain OCR page.
"""
import json
import os
import select
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import extract as extractor
import isolation

LIMIT = 25 * 1024 * 1024
# #1864/#1867: a request for a page range of a PDF (X-Document-Pages: first-last) may carry a larger file:
# only the range is converted, so the cost is the range's pages, not the file's bytes. The body is streamed
# to the temp file in CHUNK pieces, never held whole. 96 MiB matches noevia-core's
# PROJECT_DOCUMENT_READ_BYTES default and fits the /tmp tmpfs; DOCLING_RANGED_MAX_BYTES overrides it.
# /health advertises `ranges` and this cap, so a core from before this never sends a range.
CHUNK = 1024 * 1024
MAX_RANGE_PAGES = 200
# The body has 30 s per read and, in all, 30 s plus one second per BODY_RATE bytes.
BODY_SECONDS = 30
BODY_RATE = 256 * 1024


def _bytes(name, default):
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


RANGED_LIMIT = _bytes("DOCLING_RANGED_MAX_BYTES", 96 * 1024 * 1024)


def page_range(raw):
    """(first, last) from an X-Document-Pages value "first-last"; None when absent; ValueError when invalid."""
    value = (raw or "").strip(" \t")
    if value == "":
        return None
    first, sep, last = value.partition("-")
    # ASCII digits only, at most six each (str.isdigit would also take other scripts' digits).
    digits = lambda part: 0 < len(part) <= 6 and all("0" <= c <= "9" for c in part)
    if not sep or not digits(first) or not digits(last):
        raise ValueError()
    first, last = int(first), int(last)
    if not 1 <= first <= last or last - first >= MAX_RANGE_PAGES:
        raise ValueError()
    return first, last


def receive(rfile, out, size, connection):
    """Copy exactly `size` body bytes to `out` in CHUNK pieces; False when the body ends early or stalls."""
    deadline = time.monotonic() + BODY_SECONDS + size // BODY_RATE
    received = 0
    try:
        while received < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            connection.settimeout(min(BODY_SECONDS, remaining))
            chunk = rfile.read1(min(CHUNK, size - received))
            if not chunk:
                return False
            out.write(chunk)
            received += len(chunk)
    except OSError:
        return False
    return True


def _seconds(name, default):
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# Per-document wall-clock limits (#854), enforced by killing the conversion
# process, so they hold for every input type. Docling's own `document_timeout`
# only covers its PDF pipelines and, when it fires, returns a PARTIAL result
# whose missing pages would be reported as `blank` on a document shown as
# ready; a hard kill that fails the document cleanly is the safer shape.
#
# PDFs and TIFFs go through the per-page pipeline and are bounded by PAGE_CAP:
# measured worst case is 300 pages x 10.9 s = ~55 min. The limit sits under the
# web client's 65 min abort (apps/web/server/docling.cjs) so the client gets a
# 422 answer instead of giving up on its own. Everything else (Office, ODF,
# HTML, Markdown, CSV, EPUB, single images) was measured in seconds; ten
# minutes is generous, and it is what stops a dense 25 MB spreadsheet from
# holding the only slot for an hour.
LONG_SUFFIXES = {".pdf", ".tif", ".tiff"}
LONG_TIMEOUT = _seconds("DOCLING_DOCUMENT_TIMEOUT_SECONDS", 3600.0)
SHORT_TIMEOUT = _seconds("DOCLING_OTHER_TIMEOUT_SECONDS", 600.0)

# The conversion runs in a child process the server can kill (see isolation.py).
runner = isolation.IsolatedExtractor("extract:extract")
# One at a time, like the OCR worker. Docling is CPU-bound and holds its models
# resident; two concurrent conversions double the peak memory for no
# throughput on the hardware this targets.
slot = threading.BoundedSemaphore(1)


def timeout_for(suffix):
    return LONG_TIMEOUT if suffix in LONG_SUFFIXES else SHORT_TIMEOUT


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Never log document contents, names, or credentials.

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def client_gone(self):
        """True once the requester has closed its end (it stopped waiting for us).

        By now the whole request body has been read, so the socket should be
        silent until we reply; if it becomes readable and a peek returns no
        bytes, that is EOF.

        Limitation: EOF cannot tell a closed connection from a client that only
        half-closed (shutdown(SHUT_WR) after sending the body) and is still waiting
        for the answer; that client's conversion is cancelled too. The web client
        (undici fetch) never half-closes, so it is not affected.
        """
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def do_GET(self):
        if self.path == "/health":
            # `ranges` (#1864): this worker reads X-Document-Pages; noevia-core asks before it sends a range.
            return self.reply(200, {"service": "docling", "formats": sorted(extractor.SUPPORTED), "ranges": True,
                                    "rangedMaxBytes": RANGED_LIMIT})
        self.reply(404, {"service": "docling"})

    def do_POST(self):
        if self.path != "/extract":
            return self.reply(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            name = self.headers.get("X-Document-Name", "")
            # #1726: multimodal RAG asks for the figures too (noevia-core docling.cjs).
            figures = self.headers.get("X-Document-Figures", "") == "1"
            # #1864: a page range, PDFs only.
            pages = page_range(self.headers.get("X-Document-Pages"))
            if pages and not name.lower().endswith(".pdf"):
                raise ValueError()
            # The name decides the parser, so it is validated like a name and
            # never used as a path: only the suffix is read, and the bytes are
            # written to a generated temp filename.
            if not 0 < size <= (RANGED_LIMIT if pages else LIMIT) or not name or len(name) > 200 or "/" in name or "\\" in name:
                raise ValueError()
        except (ValueError, TypeError):
            return self.reply(400, {"error": "Invalid document size or name."})
        if not slot.acquire(blocking=False):
            return self.reply(503, {"error": "Document extraction is busy; refresh to retry."})
        try:
            outcome = self.convert(size, name, figures, pages)
        finally:
            # Released BEFORE the answer is written: a client that retries the
            # instant it reads a 422 must find the slot free, not a 503.
            slot.release()
        if outcome:
            self.reply(*outcome)

    def convert(self, size, name, figures=False, pages=None):
        """Read the body and convert it; returns (status, body), or None when nobody is left to answer."""
        self.connection.settimeout(30)
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        with tempfile.TemporaryDirectory(prefix="noevia-docling-") as directory:
            path = Path(directory) / ("input" + suffix)
            # Streamed to the temp file a chunk at a time (#1864), so a large ranged body is never held whole.
            with open(path, "wb") as out:
                complete = receive(self.rfile, out, size, self.connection)
            self.connection.settimeout(30)
            if not complete:
                return 400, {"error": "Incomplete document body."}
            if suffix not in extractor.SUPPORTED:
                return 415, {"error": f"This worker cannot read {suffix or 'files without an extension'}."}
            extra = {**({"figures": True} if figures else {}), **({"pages": pages} if pages else {})}
            try:
                return 200, runner.run(str(path), name, timeout_for(suffix), self.client_gone, **extra)
            except isolation.ClientGone:
                # Nobody is listening; the conversion process is already killed.
                return None
            except isolation.DocumentTimeout:
                return 422, {"error": "This document took longer than its processing time limit to read; the original is retained."}
            except isolation.WorkerUnavailable:
                # The conversion process could not start: our problem, not the document's. A 503 is
                # retried by the web client; a 422 would be cached there as permanently unreadable.
                return 503, {"error": "Document extraction is unavailable; refresh to retry."}
            except Exception:
                # Generic on purpose: the exception text can quote document
                # content, and this worker must not emit that anywhere.
                return 422, {"error": "This document could not be read within its processing limits; the original is retained."}


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("DOCLING_PORT", "8031"))), Handler).serve_forever()
