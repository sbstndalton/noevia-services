"""DOCX_TEXT_IMPL switch (#981): python stays the default and never spawns anything; rust runs the
docx-text binary and FAILS CLOSED on every failure, and the HTTP reply stays the generic 422 with
no internals. A fake binary (a small script) stands in for docx-text; the real one is covered by
test_docx_text_differential.py. Synthetic documents only."""
import http.client
import io
import json
import os
import stat
import sys
import textwrap
import threading
import unittest
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import docx_text
import server
from docx_text import SCOPE, W, extract_docx, extract_docx_py


def fixture(body):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('word/document.xml', f'<w:document xmlns:w="{W[1:-1]}"><w:body>{body}</w:body></w:document>')
    return out.getvalue()


DOC = fixture('<w:p><w:r><w:t>Synthetic switch text</w:t></w:r></w:p>')
GENERIC = 'DOCX is malformed, encrypted or exceeds its processing limits; the original is retained.'


class SwitchTests(unittest.TestCase):
    def setUp(self):
        docx_text._LOGGED.clear()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        for k in ('DOCX_TEXT_IMPL', 'DOCX_TEXT_BIN'):
            os.environ.pop(k, None)

    def fake_bin(self, body):
        """An executable script standing in for docx-text: it records argv, its environment and
        the stdin size, then runs `body` with `data` = stdin bytes."""
        tmp = Path(self.tmp.name)
        body_file = tmp / 'fake_docx_text.py'
        body_file.write_text(textwrap.dedent(f"""\
            import json, os, sys, time
            data = sys.stdin.buffer.read()
            open({str(tmp / 'calls.log')!r}, 'a').write(json.dumps([sys.argv[1:], sorted(os.environ), len(data)]) + '\\n')
        """) + textwrap.dedent(body))
        script = tmp / 'fake-docx-text'
        script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{body_file}" "$@"\n')
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return str(script)

    def calls(self):
        log = Path(self.tmp.name) / 'calls.log'
        return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []

    def use_rust(self, binary):
        os.environ['DOCX_TEXT_IMPL'] = 'rust'
        os.environ['DOCX_TEXT_BIN'] = binary

    def test_default_is_python_and_never_spawns(self):
        os.environ['DOCX_TEXT_BIN'] = self.fake_bin('sys.exit(9)')
        self.assertEqual(docx_text.impl_choice(), 'python')
        self.assertEqual(extract_docx(DOC), extract_docx_py(DOC))
        self.assertEqual(extract_docx(DOC)['text'], 'Synthetic switch text')
        self.assertEqual(self.calls(), [])

    def test_unknown_setting_is_python(self):
        os.environ['DOCX_TEXT_IMPL'] = 'fortran'
        os.environ['DOCX_TEXT_BIN'] = self.fake_bin('sys.exit(9)')
        self.assertEqual(extract_docx(DOC), extract_docx_py(DOC))
        self.assertEqual(self.calls(), [])

    def test_rust_result_is_used_with_a_minimal_environment(self):
        self.use_rust(self.fake_bin(f'print(json.dumps({{"text": "from rust", "truncated": False, "scope": {SCOPE!r}}}))'))
        os.environ['NOEVIA_SYNTHETIC_SECRET'] = 'must-not-leak'
        self.assertEqual(extract_docx(DOC), {'text': 'from rust', 'truncated': False, 'scope': SCOPE})
        [(argv, env, size)] = self.calls()
        self.assertEqual(argv, ['extract'])
        self.assertEqual(size, len(DOC))
        self.assertNotIn('NOEVIA_SYNTHETIC_SECRET', env)
        self.assertLessEqual(set(env) - {'PWD', 'SHLVL', '_', 'OLDPWD', '__CF_USER_TEXT_ENCODING', 'LC_CTYPE'}, {'PATH'})

    def test_refusal_is_a_value_error(self):
        self.use_rust(self.fake_bin('sys.stderr.write("docx-text: refused: container\\n"); sys.exit(1)'))
        with self.assertRaises(ValueError):
            extract_docx(DOC)

    def test_every_failure_fails_closed(self):
        cases = {
            'missing': None,
            'crash': 'sys.exit(3)',
            'signal': 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)',
            'not json': 'print("nope")',
            'wrong keys': 'print(json.dumps({"text": "x", "truncated": False}))',
            'extra key': f'print(json.dumps({{"text": "x", "truncated": False, "scope": {SCOPE!r}, "x": 1}}))',
            'wrong scope': 'print(json.dumps({"text": "x", "truncated": False, "scope": "other"}))',
            'bool as int': f'print(json.dumps({{"text": "x", "truncated": 0, "scope": {SCOPE!r}}}))',
            'text over budget': f'print(json.dumps({{"text": "x" * 200001, "truncated": True, "scope": {SCOPE!r}}}))',
            'output flood': 'sys.stdout.write("x" * (4 * 1024 * 1024)); sys.stdout.flush(); time.sleep(30)',
        }
        for name, body in cases.items():
            with self.subTest(name):
                self.use_rust(str(Path(self.tmp.name) / 'absent') if body is None else self.fake_bin(body))
                with self.assertRaises(docx_text.DocxTextError):
                    extract_docx(DOC)

    def test_timeout_fails_closed(self):
        self.use_rust(self.fake_bin('time.sleep(30)'))
        with patch.object(docx_text, 'DOCX_TEXT_TIMEOUT_S', 0.5):
            with self.assertRaises(docx_text.DocxTextError) as e:
                extract_docx(DOC)
        self.assertEqual(str(e.exception), 'docx-text timeout')

    def test_oversized_input_is_refused_before_spawning(self):
        self.use_rust(self.fake_bin('print("{}")'))
        with patch.object(docx_text, 'DOCX_TEXT_STDIN_CAP', 10):
            with self.assertRaises(docx_text.DocxTextError):
                extract_docx(DOC)
        self.assertEqual(self.calls(), [])

    def test_http_reply_is_the_generic_422_without_internals(self):
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)

        def post():
            c = http.client.HTTPConnection('127.0.0.1', httpd.server_address[1], timeout=30)
            c.request('POST', '/extract-docx', body=DOC, headers={'Content-Length': str(len(DOC))})
            r = c.getresponse()
            try:
                return r.status, r.read().decode()
            finally:
                c.close()

        for body in [None, 'sys.stderr.write("docx-text: refused: xml_malformed\\n"); sys.exit(1)', 'sys.exit(3)', 'print("nope")']:
            self.use_rust(str(Path(self.tmp.name) / 'absent') if body is None else self.fake_bin(body))
            status, text = post()
            self.assertEqual(status, 422)
            self.assertEqual(json.loads(text), {'error': GENERIC})
            for leak in ('docx-text', 'xml_malformed', 'exit', 'absent', 'JSON'):
                self.assertNotIn(leak, text)
        self.use_rust(self.fake_bin(f'print(json.dumps({{"text": "ok", "truncated": False, "scope": {SCOPE!r}}}))'))
        self.assertEqual(post(), (200, json.dumps({'text': 'ok', 'truncated': False, 'scope': SCOPE})))


if __name__ == '__main__':
    unittest.main()
