"""Bounded oversized-PDF reduction inside the private document worker."""
import base64
import re
import subprocess
import tempfile
import time
from pathlib import Path

INPUT_LIMIT = 60 * 1024 * 1024
OUTPUT_LIMIT = 25 * 1024 * 1024
# One overall budget for every subprocess in a reduction. The web aborts its request at 180 s
# (apps/web/server/pdf-reduce.cjs); the worker must finish, and release the single OCR slot,
# before that, leaving headroom for base64 encoding and the response. Keep it below the web abort.
REDUCE_DEADLINE_SECONDS = 160


def _run(args, limit, deadline, **kwargs):
    """subprocess.run with min(step limit, remaining budget); an exhausted budget raises
    subprocess.TimeoutExpired, which every caller already treats as a failed step."""
    timeout = min(limit, deadline - time.monotonic())
    if timeout <= 0:
        raise subprocess.TimeoutExpired(args, 0)
    return subprocess.run(args, timeout=timeout, **kwargs)


def page_count(pdf, deadline=None):
    if deadline is None:
        deadline = time.monotonic() + REDUCE_DEADLINE_SECONDS
    result = _run(['pdfinfo', str(pdf)], 15, deadline, check=True, capture_output=True)
    info = result.stdout.decode('utf-8', errors='replace')
    pages = re.search(r'^Pages:\s+(\d+)', info, re.M)
    if re.search(r'^Encrypted:\s+yes', info, re.M) or not pages or not 0 < int(pages[1]) <= 300:
        raise ValueError('PDF is encrypted, malformed or exceeds 300 pages.')
    return int(pages[1])


def reduce_pdf(data):
    if not data.startswith(b'%PDF-') or not 0 < len(data) <= INPUT_LIMIT:
        raise ValueError('Invalid PDF or PDF exceeds the 60 MB processing limit.')
    deadline = time.monotonic() + REDUCE_DEADLINE_SECONDS
    with tempfile.TemporaryDirectory(prefix='noevia-reduce-') as root:
        original, text, output = [Path(root) / name for name in ('input.pdf', 'text.txt', 'reduced.pdf')]
        original.write_bytes(data)
        try:
            pages = page_count(original, deadline)
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError('PDF cannot be read.') from error
        # Recover native text before image recompression, without printing content.
        text_ok = False
        try:
            _run(['pdftotext', '-enc', 'UTF-8', '-layout', str(original), str(text)], 45, deadline, check=True,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            text_ok = text.exists() and 0 < text.stat().st_size <= 190000
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        try:
            _run(['gs', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite',
                  '-dCompatibilityLevel=1.6', '-dPDFSETTINGS=/ebook', '-dDetectDuplicateImages=true',
                  '-dCompressFonts=true', '-sOutputFile=' + str(output), str(original)],
                 120, deadline, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if output.exists() and 0 < output.stat().st_size <= OUTPUT_LIMIT and page_count(output, deadline) == pages:
                return {'kind': 'pdf', 'dataBase64': base64.b64encode(output.read_bytes()).decode(),
                        'note': 'Compressed PDF: images may have reduced quality and interactive features may change. The original remains on your computer.'}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        if text_ok:
            native = text.read_text(encoding='utf-8').strip()
            if native:
                note = 'Text-only PDF extraction: images, scanned-page text, layout and interactive features are omitted. The original remains on your computer.'
                return {'kind': 'text', 'text': '[' + note + ']\n\n' + native, 'note': note}
        raise ValueError('Could not reduce this PDF below 25 MB or recover a complete native-text extract within the text limit. Split the PDF or compress it locally and retry.')
