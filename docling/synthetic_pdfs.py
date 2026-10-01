"""Build small synthetic PDFs for the extraction tests (#700).

Synthetic on purpose: the failure was found on a real document, and real
documents never go into this repository. These are hand-assembled PDF 1.4
files with no dependency beyond the standard library, so the fixtures are
generated at test time and nothing binary is committed.

Page kinds:
    "picture-text"  a full-page image with real (selectable) text drawn on top
                    of it — the shape Docling's layout model can classify as one
                    picture region, whose text it then skips (#700)
    "text"          plain text, no image
    "blank"         an empty content stream
    "picture"       an image and no text at all (a scan with no text layer)

Run directly to write one of each next to the current directory:
    python synthetic_pdfs.py
"""
import zlib

# Neutral, obviously synthetic prose. Long enough to clear the native-fallback
# threshold on a single page several times over.
FILLER = (
    "Synthetic fixture paragraph {n}: the quarterly widget audit counted "
    "forty-two blue widgets and seventeen green ones in aisle {n}."
)


def _escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_ops(lines, top=760, step=14):
    ops = ["BT", "/F1 10 Tf", f"72 {top} Td", f"{step} TL"]
    for line in lines:
        ops.append(f"({_escape(line)}) Tj T*")
    ops.append("ET")
    return "\n".join(ops)


def _image_ops():
    # Draw the 8x8 grey image across the whole page.
    return "q 612 0 0 792 0 0 cm /Im1 Do Q"


def page_lines(paragraphs=12, start=1):
    return [FILLER.format(n=n) for n in range(start, start + paragraphs)]


def build(pages):
    """`pages` is a list of (kind, lines). Returns the PDF as bytes."""
    objects = {}
    # 1 catalog, 2 pages tree, 3 font, 4 image; page/content objects follow.
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    pixels = zlib.compress(bytes([180]) * 64)
    objects[4] = (b"<< /Type /XObject /Subtype /Image /Width 8 /Height 8 "
                  b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
                  b"/Length " + str(len(pixels)).encode() + b" >>\nstream\n" + pixels + b"\nendstream")
    kids = []
    number = 5
    for kind, lines in pages:
        page_obj, content_obj = number, number + 1
        number += 2
        kids.append(page_obj)
        parts = []
        if kind in ("picture-text", "picture"):
            parts.append(_image_ops())
        if kind in ("picture-text", "text"):
            parts.append(_text_ops(lines))
        stream = "\n".join(parts).encode("latin-1")
        objects[content_obj] = (b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                                + stream + b"\nendstream")
        objects[page_obj] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                             f"/Resources << /Font << /F1 3 0 R >> /XObject << /Im1 4 0 R >> >> "
                             f"/Contents {content_obj} 0 R >>").encode()
    objects[2] = (f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
                  f"/Count {len(kids)} >>").encode()
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for key in sorted(objects):
        offsets[key] = len(out)
        out += f"{key} 0 obj\n".encode() + objects[key] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for key in range(1, size):
        out += f"{offsets[key]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def write(path, pages):
    with open(path, "wb") as handle:
        handle.write(build(pages))
    return path


if __name__ == "__main__":
    write("synthetic-picture-text.pdf", [("picture-text", page_lines())])
    write("synthetic-text.pdf", [("text", page_lines())])
    write("synthetic-blank.pdf", [("blank", [])])
