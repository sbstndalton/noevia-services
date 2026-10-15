"""Private, stateless PDF OCR worker. No corpus credentials or persistent volume."""
import json
import os
from docx_text import extract_docx
from pdf_reduce import reduce_pdf, INPUT_LIMIT
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LIMIT = 25 * 1024 * 1024
slot = threading.BoundedSemaphore(1)


def _bytes(name, default):
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# #1864: /extract renders only the pages it is asked for, so the file's size is not its cost: a PDF up to
# EXTRACT_LIMIT (OCR_EXTRACT_MAX_BYTES, default 256 MiB, matching noevia-core's PROJECT_DOCUMENT_READ_BYTES
# default) is accepted there and streamed to the temp file in chunks, never held whole. With the 512 MiB
# /tmp tmpfs and the 1 GiB memory limit that leaves room for one rendered page. The other routes keep LIMIT.
EXTRACT_LIMIT = _bytes("OCR_EXTRACT_MAX_BYTES", 256 * 1024 * 1024)
CHUNK = 1024 * 1024
MAX_PAGE = 100000
# #1865: X-OCR-DPI renders finer, for the retry of a page the first pass left incomplete. The default
# (300 dpi, longest side 3500 px) is unchanged; a finer render gets a proportionally larger side and more
# time per step.
DPIS = {300: (3500, 60), 400: (4700, 120), 600: (7000, 180)}


def process(data, pages, dpi=300):
    with tempfile.TemporaryDirectory(prefix="noevia-ocr-") as directory:
        pdf = Path(directory) / "input.pdf"
        pdf.write_bytes(data)
        return process_file(pdf, directory, pages, dpi)


def process_file(pdf, directory, pages, dpi=300):
    side, step = DPIS.get(dpi, DPIS[300])
    deadline = time.monotonic() + 600
    results = []
    for page in pages:
        try:
            timeout = min(step, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("document time limit reached")
            image = Path(directory) / "page"
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
                            "-scale-to", str(side), "-r", str(dpi), "-gray", "-png", str(pdf), str(image)],
                           check=True, capture_output=True, timeout=timeout)
            timeout = min(step, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("document time limit reached")
            out = subprocess.run(["tesseract", str(image) + ".png", "stdout", "-l", "eng+deu", "--psm", "3"],
                                 check=True, capture_output=True, timeout=timeout)
            text = out.stdout.decode("utf-8", errors="replace").strip()
            results.append({"number": page, "text": text[:200000], "truncated": len(text) > 200000})
            Path(str(image) + ".png").unlink(missing_ok=True)
        except (subprocess.SubprocessError, OSError, TimeoutError):
            results.append({"number": page, "error": "OCR failed or reached its processing limit; refresh to retry."})
    return results

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
        self.reply(200 if self.path == "/health" else 404, {"service": "ocr"})

    def do_POST(self):
        if self.path not in ("/extract", "/extract-docx", "/reduce-pdf"):
            return self.reply(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            pages = [1] if self.path in ("/extract-docx", "/reduce-pdf") else json.loads(self.headers.get("X-OCR-Pages", "[]"))
            limit = INPUT_LIMIT if self.path == "/reduce-pdf" else EXTRACT_LIMIT if self.path == "/extract" else LIMIT
            if not 0 < size <= limit or not isinstance(pages, list) or not 0 < len(pages) <= 50:
                raise ValueError()
            # #1864: any page of a long document (pages past 300 are read in ranges now).
            if any(type(p) is not int or not 1 <= p <= MAX_PAGE for p in pages) or len(set(pages)) != len(pages):
                raise ValueError()
            dpi = int(self.headers.get("X-OCR-DPI", "300") or "300")
            if dpi not in DPIS:
                raise ValueError()
        except (ValueError, TypeError):
            return self.reply(400, {"error": "Invalid document size or page selection."})
        if not slot.acquire(blocking=False):
            return self.reply(503, {"error": "OCR is busy; refresh to retry."})
        try:
            self.connection.settimeout(30)
            if self.path == "/extract":
                # Streamed to the temp file a chunk at a time (#1864).
                with tempfile.TemporaryDirectory(prefix="noevia-ocr-") as directory:
                    pdf = Path(directory) / "input.pdf"
                    received = 0
                    with open(pdf, "wb") as out:
                        while received < size:
                            chunk = self.rfile.read(min(CHUNK, size - received))
                            if not chunk:
                                break
                            out.write(chunk)
                            received += len(chunk)
                    if received != size:
                        return self.reply(400, {"error": "Incomplete document body."})
                    return self.reply(200, {"pages": process_file(pdf, directory, pages, dpi)})
            data = self.rfile.read(size)
            if len(data) != size:
                return self.reply(400, {"error": "Incomplete document body."})
            if self.path == "/reduce-pdf":
                try:
                    self.reply(200, reduce_pdf(data))
                except Exception:
                    self.reply(422, {"error": "PDF could not be reduced below 25 MB or extracted within limits. Split or compress it locally and retry."})
            elif self.path == "/extract-docx":
                try:
                    self.reply(200, extract_docx(data))
                except Exception:
                    self.reply(422, {"error": "DOCX is malformed, encrypted or exceeds its processing limits; the original is retained."})
        finally:
            slot.release()

if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8030), Handler).serve_forever()
