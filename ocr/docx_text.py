"""Bounded main-body DOCX text; no extraction to disk, links, macros or rendering.

DOCX_TEXT_IMPL=rust (#981) runs the same function in the bounded Rust leaf from
sbstndalton/noevia-rs (`docx-text extract`, baked into the image at the Dockerfile's pinned
NOEVIA_RS_REF). It ships dark: the default is python, which is `extract_docx_py` below, unchanged.
The Rust path FAILS CLOSED: a refusal, missing binary, nonzero exit, timeout, oversized input or
output, or malformed output raises (the server answers its generic 422), never a fallback to
Python, and nothing about the failure reaches the client. The shared differential fixtures
(tests/fixtures/docx-text.v1.json, generated in noevia-rs from `extract_docx_py`) are the
contract between the two.
"""
import io
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import threading
import zipfile
import xml.etree.ElementTree as ET

XML_LIMIT = 8 * 1024 * 1024
TEXT_LIMIT = 200000
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
SCOPE = 'DOCX body text and tables only; page layout, images, headers, footers, comments and footnotes are not interpreted.'

def extract_docx_py(data):
    # Bound central-directory parsing before ZipFile constructs member objects.
    end = data.rfind(b'PK\x05\x06', max(0, len(data) - 65557))
    if end < 0 or len(data) < end + 22:
        raise ValueError('Invalid DOCX container')
    _, disk, central_disk, disk_count, count, size, offset, comment = struct.unpack('<4s4H2LH', data[end:end+22])
    if disk or central_disk or disk_count != count or count > 1000 or size > 2*1024*1024 or offset+size > end or end+22+comment != len(data):
        raise ValueError('DOCX container exceeds its processing limits')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 1000 or len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate or excessive DOCX members')
        if sum(m.file_size for m in members) > 64*1024*1024 or any(m.flag_bits & 1 for m in members):
            raise ValueError('Encrypted or oversized DOCX container')
        try:
            info = archive.getinfo('word/document.xml')
        except KeyError:
            raise ValueError('DOCX main document is missing') from None
        if info.file_size > XML_LIMIT or info.file_size > max(1, info.compress_size)*200:
            raise ValueError('DOCX text exceeds its decompression limit')
        with archive.open(info) as stream:
            xml = stream.read(XML_LIMIT+1)
        if len(xml) > XML_LIMIT:
            raise ValueError('DOCX text exceeds its processing limit')
    # Restrict encoding before checking declarations, avoiding UTF-16/32 bypasses.
    if b'\x00' in xml or re.search(br'<!\s*(?:DOCTYPE|ENTITY)', xml, re.I):
        raise ValueError('Unsupported DOCX XML declarations or encoding')
    root = ET.fromstring(xml)
    if root.tag != W+'document':
        raise ValueError('Unsupported DOCX document namespace')
    body = root.find(W+'body')
    if body is None:
        raise ValueError('DOCX body is missing')
    chunks, size, truncated = [], 0, False
    def emit(value):
        nonlocal size, truncated
        remaining = TEXT_LIMIT-size
        if len(value) > remaining:
            truncated = True
        if remaining > 0:
            chunks.append(value[:remaining]); size += min(remaining,len(value))
    def walk(node):
        nonlocal truncated
        if size >= TEXT_LIMIT:
            truncated = True
            return
        if node.tag in (W+'del', W+'moveFrom', W+'instrText', W+'drawing', W+'pict'):
            return
        if node.tag == W+'t':
            emit(node.text or '')
            return
        if node.tag in (W+'tab', W+'br', W+'cr'):
            emit('\t' if node.tag == W+'tab' else '\n')
            return
        for child in node:
            walk(child)
        if node.tag in (W+'p', W+'tr'):
            emit('\n')
        elif node.tag == W+'tc':
            emit('\t')
    walk(body)
    text = ''.join(chunks).strip()
    return {'text': text, 'truncated': truncated, 'scope': SCOPE}


IMPLS = ('python', 'rust')
DOCX_TEXT_TIMEOUT_S = 20.0
# The server never passes more than 25 MiB (server.LIMIT); the CLI refuses more as well.
DOCX_TEXT_STDIN_CAP = 25 * 1024 * 1024
# 200k characters, each at most 6 JSON bytes (\uXXXX), plus the scope and the keys.
DOCX_TEXT_STDOUT_CAP = TEXT_LIMIT * 6 + 4096
_STDERR_CAP = 4096
_log = logging.getLogger(__name__)
_LOGGED = set()
_LOGGED_LOCK = threading.Lock()


class DocxTextError(RuntimeError):
    """DOCX_TEXT_IMPL=rust could not produce checked text (fail closed). The message is a short
    reason code only; the server never shows it."""


def _log_once(key, message):
    with _LOGGED_LOCK:
        if key in _LOGGED:
            return
        _LOGGED.add(key)
    _log.warning(message)


def impl_choice():
    """The configured implementation: "python" (default) or "rust". Anything else is python."""
    value = os.environ.get('DOCX_TEXT_IMPL', 'python').strip().lower() or 'python'
    if value not in IMPLS:
        _log_once('invalid_setting', f'DOCX_TEXT_IMPL={value!r} is not one of {", ".join(IMPLS)}; using python')
        return 'python'
    return value


def _rust_binary():
    configured = os.environ.get('DOCX_TEXT_BIN', 'docx-text').strip() or 'docx-text'
    if os.sep in configured:
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None
    return shutil.which(configured)


def _fail(reason, detail):
    # Never the document, its names or its text: a reason code and a bounded detail.
    _log_once(f'rust:{reason}', f'docx-text failed ({reason}: {detail}); refusing the document')
    return DocxTextError(f'docx-text {reason}')


def _read_capped(stream, cap, sink, overflow):
    """Read up to cap+1 bytes into sink; set overflow and stop reading past the cap."""
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            if len(sink) + len(chunk) > cap:
                sink.extend(chunk[:cap + 1 - len(sink)])
                overflow.set()
                return
            sink.extend(chunk)
    except (OSError, ValueError):
        return


def _write_all(stream, data):
    try:
        stream.write(data)
    except (OSError, ValueError):
        pass  # The child refused early or died; its exit status says why.
    finally:
        try:
            stream.close()
        except OSError:
            pass


def extract_docx_rust(data):
    binary = _rust_binary()
    if binary is None:
        raise _fail('missing_binary', 'docx-text not found or not executable')
    if not isinstance(data, (bytes, bytearray)) or len(data) > DOCX_TEXT_STDIN_CAP:
        raise _fail('input_too_large', f'more than {DOCX_TEXT_STDIN_CAP} bytes')
    try:
        # A minimal environment: the child needs nothing of ours, only a PATH.
        proc = subprocess.Popen([binary, 'extract'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, close_fds=True,
                                env={'PATH': os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')})
    except OSError as e:
        raise _fail('spawn', type(e).__name__) from None
    out, err = bytearray(), bytearray()
    out_over, err_over = threading.Event(), threading.Event()
    threads = [threading.Thread(target=_write_all, args=(proc.stdin, bytes(data)), daemon=True),
               threading.Thread(target=_read_capped, args=(proc.stdout, DOCX_TEXT_STDOUT_CAP, out, out_over), daemon=True),
               threading.Thread(target=_read_capped, args=(proc.stderr, _STDERR_CAP, err, err_over), daemon=True)]
    for t in threads:
        t.start()
    try:
        # Stop as soon as the output overflows rather than waiting for the timeout.
        deadline = DOCX_TEXT_TIMEOUT_S
        while True:
            try:
                code = proc.wait(timeout=min(0.05, deadline))
                break
            except subprocess.TimeoutExpired:
                deadline -= 0.05
                if out_over.is_set():
                    proc.kill()
                    raise _fail('output_too_large', f'more than {DOCX_TEXT_STDOUT_CAP} bytes') from None
                if deadline <= 0:
                    proc.kill()
                    raise _fail('timeout', f'no result within {DOCX_TEXT_TIMEOUT_S:g} s') from None
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for t in threads:
            t.join(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except OSError:
                pass
    if out_over.is_set():
        raise _fail('output_too_large', f'more than {DOCX_TEXT_STDOUT_CAP} bytes')
    if code == 1:
        # A refusal: the document is malformed, encrypted or over a limit, as with Python's
        # ValueError. Expected traffic, so not logged.
        raise ValueError('DOCX refused')
    if code != 0:
        raise _fail('exit', f'exit {code}: {bytes(err[:300]).decode("utf-8", "replace").strip()}')
    try:
        result = json.loads(bytes(out))
    except ValueError:
        raise _fail('malformed_output', 'not JSON') from None
    if (not isinstance(result, dict) or set(result) != {'text', 'truncated', 'scope'}
            or not isinstance(result['text'], str) or len(result['text']) > TEXT_LIMIT
            or not isinstance(result['truncated'], bool) or result['scope'] != SCOPE):
        raise _fail('malformed_output', 'unexpected shape')
    return {'text': result['text'], 'truncated': result['truncated'], 'scope': SCOPE}


def extract_docx(data):
    """Main-body DOCX text, by the configured implementation."""
    if impl_choice() == 'rust':
        return extract_docx_rust(data)
    return extract_docx_py(data)
