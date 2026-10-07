"""Differential test (#981): the docx-text binary (sbstndalton/noevia-rs) and the Python reference
`extract_docx_py` on the shared fixture table (tests/fixtures/docx-text.v1.json, byte-identical
to noevia-rs's crates/docx-text/tests/fixtures/, which CI `cmp`s at the pinned ref).

The Python half (the reference still produces every recorded expectation) always runs. The binary
half runs only when DOCX_TEXT_BIN points at a built docx-text (CI builds it at the Dockerfile's
NOEVIA_RS_REF); skipped otherwise. There the raw texts are compared, not just their digests, and
the binary must never be more lenient than Python: it never extracts what Python refuses or
returns different text, and hand-written cases refuse with Python's class unless the fixture
records why the binary is stricter. Synthetic documents only."""
import base64
import hashlib
import json
import os
import subprocess
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from docx_text import extract_docx_py

FIXTURES = Path(__file__).parent / 'tests' / 'fixtures' / 'docx-text.v1.json'
PRODUCERS = Path(__file__).parent / 'tests' / 'fixtures' / 'docx-producers.v1.json'
BIN = os.environ.get('DOCX_TEXT_BIN', '')
MESSAGES = {
    'Invalid DOCX container': 'container',
    'DOCX container exceeds its processing limits': 'limits',
    'Duplicate or excessive DOCX members': 'members',
    'Encrypted or oversized DOCX container': 'encrypted_or_oversized',
    'DOCX main document is missing': 'missing',
    'DOCX text exceeds its decompression limit': 'decompression',
    'DOCX text exceeds its processing limit': 'xml_limit',
    'Unsupported DOCX XML declarations or encoding': 'xml_declarations',
    'Unsupported DOCX document namespace': 'namespace',
    'DOCX body is missing': 'body',
}


def classify(e):
    if isinstance(e, RecursionError):
        return 'nesting'
    if isinstance(e, ET.ParseError):
        return 'xml_malformed'
    if type(e) is ValueError and str(e) in MESSAGES:
        return MESSAGES[str(e)]
    return 'container'


def compact(text, truncated):
    if len(text) > 2000:
        return {'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'text_chars': len(text), 'truncated': truncated}
    return {'text': text, 'truncated': truncated}


def python(data):
    try:
        out = extract_docx_py(data)
    except Exception as e:  # noqa: BLE001 - every exception is a refusal (the server answers 422)
        return None, {'error': classify(e)}
    return out, compact(out['text'], out['truncated'])


def rust(data):
    proc = subprocess.run([BIN, 'extract'], input=data, capture_output=True, timeout=60, check=False,
                          env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')})
    if proc.returncode == 1:
        assert proc.stdout == b''
        return None, {'error': proc.stderr.decode().removeprefix('docx-text: refused: ').strip()}
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    return out, compact(out['text'], out['truncated'])


def cases():
    return json.loads(FIXTURES.read_text())['cases']


class Differential(unittest.TestCase):
    def test_python_reference_still_produces_every_expectation(self):
        all_cases = cases()
        self.assertGreaterEqual(sum(c['kind'] == 'case' for c in all_cases), 100)
        self.assertGreaterEqual(sum(c['kind'] == 'mutant' for c in all_cases), 500)
        for c in all_cases:
            self.assertEqual(python(base64.b64decode(c['docx']))[1], c['python'], c['name'])

    @unittest.skipUnless(BIN, 'DOCX_TEXT_BIN not set: no docx-text binary to compare')
    def test_binary_matches_and_is_never_more_lenient(self):
        counts = {'identical': 0, 'both_refused': 0, 'stricter': 0}
        for c in cases():
            data = base64.b64decode(c['docx'])
            py_raw, py = python(data)
            rs_raw, rs = rust(data)
            self.assertEqual(rs, c['rust'], c['name'])
            if rs_raw is not None:
                self.assertIsNotNone(py_raw, f"{c['name']}: binary extracted what Python refuses")
                self.assertEqual(rs_raw['text'], py_raw['text'], c['name'])
                self.assertEqual(rs_raw['truncated'], py_raw['truncated'], c['name'])
                self.assertEqual(rs_raw['scope'], py_raw['scope'], c['name'])
                counts['identical'] += 1
            elif py_raw is None:
                counts['both_refused'] += 1
            else:
                counts['stricter'] += 1
            if c['kind'] == 'case' and 'stricter' not in c:
                self.assertEqual(rs, py, f"{c['name']}: class differs from Python")
        print('docx-text differential:', counts)

    def test_python_extracts_every_producer_file(self):
        for c in json.loads(PRODUCERS.read_text())['cases']:
            self.assertEqual(python(base64.b64decode(c['docx']))[1], c['python'], c['name'])
            self.assertNotIn('error', c['python'], c['name'])

    @unittest.skipUnless(BIN, 'DOCX_TEXT_BIN not set: no docx-text binary to compare')
    def test_binary_extracts_producer_files_like_python(self):
        """python-docx, pandoc, LibreOffice, textutil and zip re-packs (synthetic content): the
        binary returns Python's exact text, except the recorded deliberate refusals."""
        for c in json.loads(PRODUCERS.read_text())['cases']:
            data = base64.b64decode(c['docx'])
            py_raw, _ = python(data)
            rs_raw, rs = rust(data)
            self.assertEqual(rs, c['rust'], c['name'])
            if 'stricter' in c:
                self.assertIsNone(rs_raw, c['name'])
            else:
                self.assertEqual(rs_raw, py_raw, c['name'])


if __name__ == '__main__':
    unittest.main()
