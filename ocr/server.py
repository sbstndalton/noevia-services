"""Private, stateless PDF OCR worker. No corpus credentials or persistent volume."""
import json
from docx_text import extract_docx
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LIMIT = 25 * 1024 * 1024
slot = threading.BoundedSemaphore(1)

def process(data, pages):
    deadline = time.monotonic() + 600
    results = []
    with tempfile.TemporaryDirectory(prefix="noevia-ocr-") as directory:
        pdf = Path(directory) / "input.pdf"
        pdf.write_bytes(data)
        for page in pages:
            try:
                timeout = min(60, deadline - time.monotonic())
                if timeout <= 0:
                    raise TimeoutError("document time limit reached")
                image = Path(directory) / "page"
                subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
                                "-scale-to", "3500", "-r", "300", "-gray", "-png", str(pdf), str(image)],
                               check=True, capture_output=True, timeout=timeout)
                timeout = min(60, deadline - time.monotonic())
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
        if self.path not in ("/extract", "/extract-docx"):
            return self.reply(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            pages = [1] if self.path == "/extract-docx" else json.loads(self.headers.get("X-OCR-Pages", "[]"))
            if not 0 < size <= LIMIT or not isinstance(pages, list) or not 0 < len(pages) <= 50:
                raise ValueError()
            if any(type(p) is not int or not 1 <= p <= 300 for p in pages) or len(set(pages)) != len(pages):
                raise ValueError()
        except (ValueError, TypeError):
            return self.reply(400, {"error": "Invalid document size or page selection."})
        if not slot.acquire(blocking=False):
            return self.reply(503, {"error": "OCR is busy; refresh to retry."})
        try:
            self.connection.settimeout(30)
            data = self.rfile.read(size)
            if len(data) != size:
                return self.reply(400, {"error": "Incomplete document body."})
            if self.path == "/extract-docx":
                try:
                    self.reply(200, extract_docx(data))
                except Exception:
                    self.reply(422, {"error": "DOCX is malformed, encrypted or exceeds its processing limits; the original is retained."})
            else:
                self.reply(200, {"pages": process(data, pages)})
        finally:
            slot.release()

if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8030), Handler).serve_forever()
