"""The fixture generator itself, which needs no OCR engine and so runs everywhere (#856).

The real-engine tests are only as good as their inputs: these pin that the generated PDFs are
well formed, that the "scan" really has no text layer, that the encrypted file really is
password-protected, and that a missing engine is named in the skip reason instead of the
tests silently passing.
"""
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import synthetic_pdfs as fixtures


def xref_offsets(data):
    start = int(re.search(rb'startxref\n(\d+)\n%%EOF\n$', data)[1])
    table = data[start:].split(b'\n')
    count = int(table[1].split()[1])
    return {n: int(table[2 + n].split()[0]) for n in range(1, count) if table[2 + n].split()[2] == b'n'}


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        cls.files = fixtures.write_all(cls._directory.name)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_every_fixture_is_written_and_small(self):
        self.assertEqual(set(self.files), {'scanned.pdf', 'mixed-page.pdf', 'mixed-pages.pdf', 'text.pdf', 'malformed.pdf', 'encrypted.pdf'})
        for path in self.files.values():
            self.assertTrue(path.read_bytes().startswith(b'%PDF-'))
            self.assertLess(path.stat().st_size, 100_000)

    def test_cross_reference_offsets_point_at_their_objects(self):
        for name in ['scanned.pdf', 'mixed-page.pdf', 'mixed-pages.pdf', 'text.pdf', 'encrypted.pdf']:
            data = self.files[name].read_bytes()
            for number, offset in xref_offsets(data).items():
                self.assertTrue(data[offset:].startswith(b'%d 0 obj\n' % number), (name, number))

    def test_the_scan_is_an_image_with_no_text_operators(self):
        data = self.files['scanned.pdf'].read_bytes()
        self.assertIn(b'/Subtype /Image', data)
        self.assertNotIn(b' Tj', data)
        self.assertNotIn(b'/Encrypt', data)

    def test_the_mixed_pages_have_native_text_beside_the_scan(self):
        self.assertIn(b'(' + fixtures.HEADER.encode() + b') Tj', self.files['mixed-page.pdf'].read_bytes())
        data = self.files['mixed-pages.pdf'].read_bytes()
        self.assertIn(b'/Count 2', data)
        self.assertIn(b'(NATIVE PAGE ONE) Tj', data)
        self.assertIn(b'/Subtype /Image', data)

    def test_the_scan_has_ink_where_the_text_is_and_none_elsewhere(self):
        width, height, pixels = fixtures.render_lines(tuple(fixtures.ROWS))
        self.assertEqual(len(pixels), width * height)
        dark = [i for i, value in enumerate(pixels) if value < 128]
        self.assertGreater(len(dark), 2000)
        rows = {i // width for i in dark}
        self.assertGreaterEqual(min(rows), 240 - 12)
        # Four lines of 7 dots at 6 px, plus the gaps: nothing is drawn below them.
        self.assertLess(max(rows), 240 + 4 * 10 * fixtures.CELL)

    def test_every_character_in_the_rows_has_a_glyph_of_the_right_shape(self):
        for char in set(''.join(fixtures.ROWS + [fixtures.HEADER])):
            glyph = fixtures._FONT[char]
            self.assertEqual(len(glyph), fixtures.GLYPH_H, char)
            self.assertTrue(all(len(row) == fixtures.GLYPH_W for row in glyph), char)
        # Only 0 and O may share a shape (see the comment on the glyph).
        shapes = [tuple(g) for char, g in fixtures._FONT.items() if char != '0']
        self.assertEqual(len(set(shapes)), len(shapes), 'two characters share a glyph')

    def test_the_encrypted_pdf_declares_standard_security_with_a_user_password(self):
        data = self.files['encrypted.pdf'].read_bytes()
        self.assertIn(b'/Encrypt', data)
        self.assertIn(b'/Filter /Standard /V 1 /R 2', data)
        entry = re.search(rb'/O <([0-9a-f]+)> /U <([0-9a-f]+)>', data)
        self.assertEqual((len(entry[1]), len(entry[2])), (64, 64))  # 32 bytes each
        # The page text is not readable without the key.
        self.assertNotIn(b'ENCRYPTED SYNTHETIC PAGE', data)

    def test_the_malformed_pdf_has_no_cross_reference_table(self):
        data = self.files['malformed.pdf'].read_bytes()
        self.assertTrue(data.startswith(b'%PDF-'))
        self.assertNotIn(b'startxref', data)

    def test_padding_makes_a_valid_oversized_file(self):
        data = fixtures.pdf([{'text': ['X']}], padding=100_000)
        self.assertGreater(len(data), 100_000)
        for number, offset in xref_offsets(data).items():
            self.assertTrue(data[offset:].startswith(b'%d 0 obj\n' % number))
        self.assertGreater(fixtures.OVERSIZE_PADDING, 25 * 1024 * 1024)


class EngineDetectionTests(unittest.TestCase):
    def test_a_missing_engine_is_named_in_the_skip_reason(self):
        with patch('synthetic_pdfs.shutil.which', side_effect=lambda tool: None if tool == 'gs' else '/usr/bin/' + tool), \
                patch('synthetic_pdfs.subprocess.run') as run:
            run.return_value.stdout = b'List of available languages (3):\neng\ndeu\nosd\n'
            self.assertEqual(fixtures.missing_engines(), ['gs'])
            self.assertIn('gs', fixtures.skip_reason())

    def test_missing_language_data_is_named_too(self):
        with patch('synthetic_pdfs.shutil.which', return_value='/usr/bin/tool'), \
                patch('synthetic_pdfs.subprocess.run') as run:
            run.return_value.stdout = b'List of available languages (2):\neng\nosd\n'
            self.assertEqual(fixtures.missing_engines(), ["tesseract language data 'deu'"])

    def test_nothing_missing_means_no_skip(self):
        with patch('synthetic_pdfs.shutil.which', return_value='/usr/bin/tool'), \
                patch('synthetic_pdfs.subprocess.run') as run:
            run.return_value.stdout = b'eng\ndeu\n'
            self.assertEqual(fixtures.skip_reason(), '')


if __name__ == '__main__':
    unittest.main()
