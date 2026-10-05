"""Build synthetic PDFs for the OCR worker's real-engine tests (#856).

Generated at test time, standard library only, nothing binary committed, and
nothing here is real data: the "financial rows" are invented numbers.

The tests that need these (OCR of a scan, Ghostscript reduction, encrypted and
malformed input) can only run where tesseract, poppler and ghostscript are
installed, so `missing_engines()` names whatever is absent and the tests skip
with that exact reason instead of passing vacuously.

Fixtures (`write_all(directory)`):
    scanned.pdf       one page that is only an image of text: no text layer, so the
                      words can only be found by OCR
    mixed-page.pdf    the same scan with a native text line ("DIGITAL HEADER") on it
    mixed-pages.pdf   page 1 native text only, page 2 the scan
    text.pdf          native text, no image (`oversized_text_pdf()` pads it past 25 MB for the
                      reduction test)
    malformed.pdf     a PDF header and then garbage
    encrypted.pdf     RC4-40 standard-security PDF with a non-empty user password, so a
                      reader without the password is refused (hand-built, no crypto library)

The scan is drawn with a 5x7 dot-matrix capital font so tesseract reads it
reliably without any imaging dependency, which is why the rows are upper case.

Run directly to write them into the current directory:
    python synthetic_pdfs.py
"""
import hashlib
import shutil
import struct
import subprocess
import zlib
from functools import lru_cache
from pathlib import Path

# What the scanned page says. Mirrors a bank-statement shape: an identifier, signed amounts and a total.
ROWS = [
    "REF-2042",
    "2042-01-02 UTILITIES 42.15",
    "2042-01-03 REFUND -7.20",
    "TOTAL 34.95",
]
HEADER = "DIGITAL HEADER"

_FONT = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", ".#.#.", ".#.#.", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    # Same shape as "O" on purpose. A zero with a diagonal or dot inside was read as 6, 4, H, B or b
    # by tesseract (measured in CI, every size and blur); the plain oval reads as 0 among digits.
    "0": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    ".": [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
    " ": ["....."] * 7,
}

# Image pixels per font dot, and blur passes. Chosen by running tesseract over a grid of sizes
# and blurs in CI: 3 px dots with one pass read all four rows exactly at every blur level tried,
# while 6 px dots only read when blurred and misread a comma for a full stop without it.
CELL = 3
BLUR_PASSES = 1
IMAGE_W, IMAGE_H = 1275, 1650   # the whole US-Letter page at 150 dpi
GLYPH_W, GLYPH_H = 5, 7


def _box_blur(rows):
    """3x3 box blur of a list of equal-length byte rows (edges keep their value)."""
    horizontal = []
    for row in rows:
        mid = bytes((a + b + c) // 3 for a, b, c in zip(row, row[1:], row[2:]))
        horizontal.append(row[:1] + mid + row[-1:])
    out = [horizontal[0]]
    for above, row, below in zip(horizontal, horizontal[1:], horizontal[2:]):
        out.append(bytes((a + b + c) // 3 for a, b, c in zip(above, row, below)))
    out.append(horizontal[-1])
    return out


@lru_cache(maxsize=None)
def render_lines(lines, left=96, top=240, line_gap=3):
    """Gray bitmap (white page, text in dots) of `lines`; returns (width, height, bytes).

    `lines` is a tuple (hashable, so the same scan is built once per test run). The dots
    are softened with a 3x3 blur: a raw 5x7 dot matrix is read as junk by some engines
    (raw dots came back with digits and punctuation wrong), the softened strokes read cleanly, and a real
    scan is never pixel-perfect anyway.
    """
    page = [bytearray(b"\xff" * IMAGE_W) for _ in range(IMAGE_H)]
    advance = (GLYPH_W + 1) * CELL
    for row, text in enumerate(lines):
        y0 = top + row * (GLYPH_H + line_gap) * CELL
        for column, char in enumerate(text):
            x0 = left + column * advance
            assert x0 + advance <= IMAGE_W and y0 + GLYPH_H * CELL <= IMAGE_H, "text does not fit the page"
            for gy, glyph_row in enumerate(_FONT[char]):
                for gx, dot in enumerate(glyph_row):
                    if dot != "#":
                        continue
                    for dy in range(CELL):
                        start = x0 + gx * CELL
                        page[y0 + gy * CELL + dy][start:start + CELL] = b"\x00" * CELL
    first = top - 2 * CELL
    last = top + len(lines) * (GLYPH_H + line_gap) * CELL + 2 * CELL
    region = [bytes(r) for r in page[first:last]]
    for _ in range(BLUR_PASSES):
        region = _box_blur(region)
    page[first:last] = region
    return IMAGE_W, IMAGE_H, b"".join(bytes(r) for r in page)


def _escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _stream(dictionary, data):
    return b"<< " + dictionary + b" /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream"


def build(pages, encrypt=None):
    """Page objects for a PDF 1.4 file. `pages` is a list of dicts
    {"scan": [lines] or None, "text": [lines] or None}; `encrypt`, when given, is
    (object_number, bytes) -> bytes applied to each content stream."""
    objects = {3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids = []
    number = 4
    for page in pages:
        page_obj, content_obj = number, number + 1
        number += 2
        kids.append(page_obj)
        ops, resources = [], b"/Font << /F1 3 0 R >>"
        if page.get("scan"):
            image_obj = number
            number += 1
            width, height, pixels = render_lines(tuple(page["scan"]))
            objects[image_obj] = _stream(
                b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
                b"/BitsPerComponent 8 /Filter /FlateDecode" % (width, height), zlib.compress(pixels, 9))
            resources += b" /XObject << /Im1 %d 0 R >>" % image_obj
            ops.append("q 612 0 0 792 0 0 cm /Im1 Do Q")
        if page.get("text"):
            ops.append("BT /F1 18 Tf 72 740 Td 24 TL")
            ops.extend(f"({_escape(line)}) Tj T*" for line in page["text"])
            ops.append("ET")
        content = "\n".join(ops).encode("latin-1")
        if encrypt:
            content = encrypt(content_obj, content)
        objects[content_obj] = _stream(b"", content)
        objects[page_obj] = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << "
                             + resources + b" >> /Contents %d 0 R >>" % content_obj)
    objects[2] = b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % k for k in kids) + b"] /Count %d >>" % len(kids)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    return objects


def _serialise(objects, trailer=b"", padding=0):
    """`padding` > 0 adds a comment line of that many bytes after the header: a valid but
    oversized file whose xref offsets still match (the trailer stays at the end)."""
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    if padding:
        out += b"%" + b" " * (padding - 2) + b"\n"
    offsets = {}
    for key in sorted(objects):
        offsets[key] = len(out)
        out += b"%d 0 obj\n" % key + objects[key] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for key in range(1, size):
        out += b"%010d 00000 n \n" % offsets[key]
    out += b"trailer\n<< /Size %d /Root 1 0 R %s >>\nstartxref\n%d\n%%%%EOF\n" % (size, trailer, xref)
    return bytes(out)


def pdf(pages, padding=0):
    return _serialise(build(pages), padding=padding)


# -- encrypted ---------------------------------------------------------------
_PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


def _rc4(key, data):
    state = list(range(256))
    j = 0
    for i in range(256):
        j = (j + state[i] + key[i % len(key)]) & 255
        state[i], state[j] = state[j], state[i]
    out, i, j = bytearray(), 0, 0
    for byte in data:
        i = (i + 1) & 255
        j = (j + state[i]) & 255
        state[i], state[j] = state[j], state[i]
        out.append(byte ^ state[(state[i] + state[j]) & 255])
    return bytes(out)


def encrypted_pdf(user_password=b"synthetic-user-pw", owner_password=b"synthetic-owner-pw"):
    """Standard security handler, revision 2 (RC4, 40-bit), PDF 32000-1 section 7.6.3.

    The user password is NOT empty: a reader that is not given one (pdfinfo, pdftoppm,
    the OCR worker) must be refused, which is what makes this a real "encrypted" input.
    """
    document_id = hashlib.md5(b"noevia synthetic fixture").digest()
    permissions = -4 & 0xFFFFFFFF
    pad = lambda password: (password + _PAD)[:32]
    owner = _rc4(hashlib.md5(pad(owner_password)).digest()[:5], pad(user_password))
    key = hashlib.md5(pad(user_password) + owner + struct.pack("<I", permissions) + document_id).digest()[:5]
    user = _rc4(key, _PAD)

    def encrypt(object_number, data):
        object_key = hashlib.md5(key + struct.pack("<I", object_number)[:3] + b"\x00\x00").digest()[:10]
        return _rc4(object_key, data)

    objects = build([{"text": ["ENCRYPTED SYNTHETIC PAGE"]}], encrypt=encrypt)
    encrypt_obj = max(objects) + 1
    objects[encrypt_obj] = (b"<< /Filter /Standard /V 1 /R 2 /P %d /O <%s> /U <%s> >>"
                            % (-4, owner.hex().encode(), user.hex().encode()))
    trailer = b"/Encrypt %d 0 R /ID [<%s> <%s>]" % (encrypt_obj, document_id.hex().encode(), document_id.hex().encode())
    return _serialise(objects, trailer)


def malformed_pdf():
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 99 0 R\nthis file stops mid-object"


# -- fixture set -------------------------------------------------------------
OVERSIZE_PADDING = 26 * 1024 * 1024  # `pdf(..., padding=)` this much to pass the 25 MB limit pdf_reduce must get under


def oversized_text_pdf():
    """A valid PDF whose only content is one native text line, padded past 25 MB."""
    return pdf([{"text": ["SYNTHETIC NATIVE TEXT FOR REDUCTION"]}], padding=OVERSIZE_PADDING)


def write_all(directory):
    """Write every fixture into `directory` (created if needed); returns {name: Path}."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "scanned.pdf": pdf([{"scan": ROWS}]),
        "mixed-page.pdf": pdf([{"scan": ROWS, "text": [HEADER]}]),
        "mixed-pages.pdf": pdf([{"text": ["NATIVE PAGE ONE"]}, {"scan": ROWS}]),
        "text.pdf": pdf([{"text": ["SYNTHETIC NATIVE TEXT FOR REDUCTION"]}]),
        "malformed.pdf": malformed_pdf(),
        "encrypted.pdf": encrypted_pdf(),
    }
    paths = {}
    for name, data in files.items():
        paths[name] = directory / name
        paths[name].write_bytes(data)
    return paths


def missing_engines():
    """Names of the real engines the OCR tests need that are not usable here (empty = all present)."""
    missing = [tool for tool in ("tesseract", "pdftoppm", "pdfinfo", "pdftotext", "gs") if shutil.which(tool) is None]
    if "tesseract" not in missing:
        try:
            langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True, timeout=30, check=True)
            installed = set(langs.stdout.decode(errors="replace").split())
        except (OSError, subprocess.SubprocessError):
            installed = set()
        missing += [f"tesseract language data '{lang}'" for lang in ("eng", "deu") if lang not in installed]
    return missing


def skip_reason():
    missing = missing_engines()
    return "real-engine OCR/PDF tests need what is not installed here: " + ", ".join(missing) if missing else ""


if __name__ == "__main__":
    for name, path in write_all(".").items():
        print(path, path.stat().st_size)
