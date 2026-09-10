"""Bounded main-body DOCX text; no extraction to disk, links, macros or rendering."""
import io
import re
import struct
import zipfile
import xml.etree.ElementTree as ET

XML_LIMIT = 8 * 1024 * 1024
TEXT_LIMIT = 200000
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
SCOPE = 'DOCX body text and tables only; page layout, images, headers, footers, comments and footnotes are not interpreted.'

def extract_docx(data):
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
