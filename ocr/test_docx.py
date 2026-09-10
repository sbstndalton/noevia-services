import io
import unittest
import zipfile
from docx_text import extract_docx, W

def fixture(body, extras=None):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('word/document.xml',f'<w:document xmlns:w="{W[1:-1]}"><w:body>{body}</w:body></w:document>')
        for name,text in (extras or {}).items(): z.writestr(name,text)
    return out.getvalue()

class DocxTests(unittest.TestCase):
    def test_body_table_hyperlink_text_and_insertions_preserve_order_without_hidden_content(self):
        data=fixture('<w:p><w:r><w:t>Synthetic invoice</w:t></w:r><w:hyperlink><w:r><w:t> linked label</w:t></w:r></w:hyperlink></w:p>'
                     '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Refund</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>-7.20</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
                     '<w:p><w:del><w:r><w:delText>DELETED SECRET</w:delText></w:r></w:del><w:ins><w:r><w:t>Accepted insertion</w:t></w:r></w:ins><w:r><w:instrText>HYPERLINK http://never-fetch.invalid</w:instrText></w:r></w:p>',
                     {'word/header1.xml':'DO NOT INCLUDE HEADER','word/vbaProject.bin':'NEVER EXECUTE'})
        out=extract_docx(data)
        self.assertIn('Synthetic invoice linked label',out['text'])
        self.assertLess(out['text'].index('Refund'),out['text'].index('-7.20'))
        self.assertIn('Accepted insertion',out['text'])
        for text in ['DELETED','never-fetch','HEADER','EXECUTE']: self.assertNotIn(text,out['text'])
        self.assertIn('body text',out['scope']);self.assertFalse(out['truncated'])

    def test_missing_main_document_and_invalid_archive_are_rejected(self):
        for data in [b'not a zip',fixture('<w:p/>')[:-12]]:
            with self.assertRaises(Exception): extract_docx(data)
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z: z.writestr('word/other.xml','irrelevant')
        with self.assertRaisesRegex(ValueError,'missing'):extract_docx(out.getvalue())

    def test_entities_and_utf16_declarations_are_rejected(self):
        for xml in ['<!DOCTYPE x [<!ENTITY x "EXPANDED">]><w:document xmlns:w="'+W[1:-1]+'"><w:body>&x;</w:body></w:document>', '<!DOCTYPE x><x/>'.encode('utf-16')]:
            out=io.BytesIO()
            with zipfile.ZipFile(out,'w') as z: z.writestr('word/document.xml',xml)
            with self.assertRaisesRegex(ValueError,'declarations or encoding'):extract_docx(out.getvalue())

    def test_duplicate_member_and_decompression_bomb_are_rejected(self):
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            z.writestr('word/document.xml','one')
            with self.assertWarns(UserWarning):z.writestr('word/document.xml','two')
        with self.assertRaisesRegex(ValueError,'Duplicate'):extract_docx(out.getvalue())
        with self.assertRaisesRegex(ValueError,'decompression'):extract_docx(fixture('<w:p><w:r><w:t>'+'A'*500000+'</w:t></w:r></w:p>'))

    def test_text_budget_is_bounded_and_reported(self):
        import random,string
        rng=random.Random(42)
        text=''.join(rng.choices(string.ascii_letters,k=220000))
        out=extract_docx(fixture('<w:p><w:r><w:t>'+text+'</w:t></w:r></w:p>'))
        self.assertEqual(len(out['text']),200000);self.assertTrue(out['truncated'])

if __name__=='__main__':unittest.main()
