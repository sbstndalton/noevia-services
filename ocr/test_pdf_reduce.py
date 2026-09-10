import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from pdf_reduce import reduce_pdf

class ReductionTests(unittest.TestCase):
    def test_invalid_pdf_is_rejected(self):
        with self.assertRaises(ValueError): reduce_pdf(b'not a PDF')

    def test_failed_compression_uses_labelled_native_text_and_cleans_files(self):
        import tempfile
        def run(args, **kwargs):
            if args[0]=='pdfinfo':return subprocess.CompletedProcess(args,0,stdout=b'Pages: 1\nEncrypted: no\n')
            if args[0]=='pdftotext': Path(args[-1]).write_text('Synthetic full text');return
            raise subprocess.CalledProcessError(1,args)
        with tempfile.TemporaryDirectory() as root, patch('tempfile.tempdir',root), patch('pdf_reduce.subprocess.run',side_effect=run):
            result=reduce_pdf(b'%PDF-synthetic')
            self.assertEqual(result['kind'],'text');self.assertIn('omitted',result['text'])
            self.assertEqual(list(Path(root).iterdir()),[])

    def test_empty_or_excessive_text_does_not_become_a_silent_partial_source(self):
        def run(args, **kwargs):
            if args[0]=='pdfinfo':return subprocess.CompletedProcess(args,0,stdout=b'Pages: 1\nEncrypted: no\n')
            if args[0]=='pdftotext': Path(args[-1]).write_text('x'*200001);return
            raise subprocess.CalledProcessError(1,args)
        with patch('pdf_reduce.subprocess.run',side_effect=run), self.assertRaises(ValueError): reduce_pdf(b'%PDF-synthetic')

    @unittest.skipUnless(Path('/fixtures/text.pdf').exists(), 'mount synthetic PDFs')
    def test_real_oversized_pdf_is_reduced_and_keeps_native_text(self):
        import base64,tempfile
        data=Path('/fixtures/text.pdf').read_bytes()+b'\n%'+b' '* (26*1024*1024)
        result=reduce_pdf(data);self.assertEqual(result['kind'],'pdf')
        output=base64.b64decode(result['dataBase64']);self.assertLess(len(output),25*1024*1024)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'out.pdf';p.write_bytes(output)
            text=subprocess.check_output(['pdftotext',str(p),'-']).decode();self.assertTrue(text.strip())

    @unittest.skipUnless(Path('/fixtures/encrypted.pdf').exists(), 'mount synthetic PDFs')
    def test_real_encrypted_and_malformed_pdf_fail(self):
        for name in ['encrypted.pdf','malformed.pdf']:
            with self.subTest(name=name),self.assertRaises(ValueError):reduce_pdf(Path('/fixtures',name).read_bytes())
