"""Run in the OCR image with synthetic fixtures mounted at /fixtures."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch
import server

class WorkerTests(unittest.TestCase):
    def test_subprocess_failure_is_explicit_and_temporary_files_are_removed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            with patch('tempfile.tempdir', root), patch('subprocess.run', side_effect=OSError('synthetic failure')):
                out = server.process(b'fake', [1])
            self.assertIn('error', out[0])
            self.assertEqual(list(Path(root).iterdir()), [])

    @unittest.skipUnless(Path('/fixtures/scanned.pdf').exists(), 'mount synthetic fixtures for real OCR')
    def test_real_scans_preserve_financial_rows_and_signs(self):
        for name, page in [('scanned.pdf', 1), ('mixed-page.pdf', 1), ('mixed-pages.pdf', 2)]:
            with self.subTest(name=name):
                out = server.process(Path('/fixtures', name).read_bytes(), [page])[0]
                self.assertNotIn('error', out)
                text = out['text']
                for value in ['INV-2042', '2042-01-02 Utilities 42.15', '2042-01-03 Refund -7.20', 'TOTAL 34.95']:
                    self.assertIn(value, text)
                if name == 'mixed-page.pdf':
                    self.assertIn('DIGITAL HEADER', text)

    @unittest.skipUnless(Path('/fixtures/malformed.pdf').exists(), 'mount synthetic fixtures for real OCR')
    def test_real_invalid_and_encrypted_documents_fail_cleanly(self):
        for name in ['malformed.pdf', 'encrypted.pdf']:
            self.assertIn('error', server.process(Path('/fixtures', name).read_bytes(), [1])[0])

if __name__ == '__main__':
    unittest.main()
