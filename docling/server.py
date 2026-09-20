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
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import extract as extractor

LIMIT = 25 * 1024 * 1024
# One at a time, like the OCR worker. Docling is CPU-bound and holds its models
# resident; two concurrent conversions double the peak memory for no
# throughput on the hardware this targets.
slot = threading.BoundedSemaphore(1)


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

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"service": "docling", "formats": sorted(extractor.SUPPORTED)})
        self.reply(404, {"service": "docling"})

    def do_POST(self):
        if self.path != "/extract":
            return self.reply(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            name = self.headers.get("X-Document-Name", "")
            # The name decides the parser, so it is validated like a name and
            # never used as a path: only the suffix is read, and the bytes are
            # written to a generated temp filename.
            if not 0 < size <= LIMIT or not name or len(name) > 200 or "/" in name or "\\" in name:
                raise ValueError()
        except (ValueError, TypeError):
            return self.reply(400, {"error": "Invalid document size or name."})
        if not slot.acquire(blocking=False):
            return self.reply(503, {"error": "Document extraction is busy; refresh to retry."})
        try:
            self.connection.settimeout(30)
            data = self.rfile.read(size)
            if len(data) != size:
                return self.reply(400, {"error": "Incomplete document body."})
            suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if suffix not in extractor.SUPPORTED:
                return self.reply(415, {"error": f"This worker cannot read {suffix or 'files without an extension'}."})
            with tempfile.TemporaryDirectory(prefix="noevia-docling-") as directory:
                path = Path(directory) / ("input" + suffix)
                path.write_bytes(data)
                try:
                    self.reply(200, extractor.extract(str(path), name))
                except Exception:
                    # Generic on purpose: the exception text can quote document
                    # content, and this worker must not emit that anywhere.
                    self.reply(422, {"error": "This document could not be read within its processing limits; the original is retained."})
        finally:
            slot.release()


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("DOCLING_PORT", "8031"))), Handler).serve_forever()
