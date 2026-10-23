"""Docling-backed document extraction.

Everything that touches Docling's API lives in `_convert` and `_pages_from`.
That is deliberate: this file is written against Docling's documented API but
was NOT executed against a real Docling install during development (the build
environment could not reach HuggingFace to fetch the models), so if the API has
moved, two functions need changing and nothing else does. Run `selftest.py`
against a real PDF before trusting it.

What this replaces: a pdf.js text walk that emitted items in raw order with no
x/y sort, so multi-column pages interleaved, and that understood exactly one
format. Docling brings reading order, table structure, and the Office/ODF
formats that were previously accepted and then silently dropped.

Output contract is the page list apps/web/server/documents.cjs already builds:
    {"number": int, "text": str, "status": str, "method": str, "truncated": bool}
`status` is one of native | blank | truncated | degraded. A `degraded` page
also carries `"reason": "native-fallback"`: Docling returned (almost) nothing
for it, but the PDF's own text layer had real text, so that text was used
instead (#700). The caller's vocabulary also
has ocr/unreadable/failed, which this extractor never emits: Docling runs OCR
inline rather than as a separate pass, and it cannot distinguish an
unreadable page from an empty one.
"""

import os
from collections import namedtuple

# What `_convert` hands back. A plain tuple rather than an attribute stashed on
# the document: DoclingDocument is a pydantic model with strict fields, so
# `setattr(document, "source_page_count", n)` raises ValueError("DoclingDocument
# object has no field ...") — and it raises only AFTER the conversion has been
# paid for, which on a 334-page PDF is 454 seconds of wasted work.
Converted = namedtuple("Converted", "document page_count")

PAGE_TEXT_CAP = 200_000
TOTAL_TEXT_CAP = 2_000_000
PAGE_CAP = 300
# Docling's own admission limits (#854), passed to convert(). The server already
# refuses bodies over 25 MB, so FILE_SIZE_CAP is defence in depth. The page
# limit applies to non-PDF inputs only: Docling counts sheets, slides and
# frames as pages there and, unlike `page_range` (which only trims what is
# converted), a document over `max_num_pages` is REJECTED before any work is
# done. PDFs keep today's behaviour of converting the first PAGE_CAP pages of
# any length of book. The wall-clock limit lives in the native server (noevia-docling), which kills the worker.
FILE_SIZE_CAP = 25 * 1024 * 1024
NON_PDF_MAX_PAGES = 1000
# #1864: a page-range conversion of a PDF (the native server's X-Document-Pages) may come from a larger file; only
# the range is converted. Defence in depth behind the native server's ranged cap (same variable, same default;
# assigned below, once _env_int exists).
RANGED_FILE_SIZE_CAP = 96 * 1024 * 1024


def _env_int(name, default):
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# Native-text fallback (#700). A PDF page whose text layer holds at least this
# many non-whitespace characters is not allowed to come back empty: measured in
# the #262 verification, one page with ~5k characters of native text arrived
# from Docling with nothing, was reported `blank`, and the document showed as
# ready. 200 is roughly two short sentences — well above the page numbers,
# running heads and stray glyphs a picture-only page can carry in its text
# layer, well below a page of prose.
NATIVE_FALLBACK_MIN_CHARS = _env_int("DOCLING_NATIVE_FALLBACK_MIN_CHARS", 200)
RANGED_FILE_SIZE_CAP = _env_int("DOCLING_RANGED_MAX_BYTES", RANGED_FILE_SIZE_CAP)
# "Near-empty": Docling kept less than this share of the native text. Docling
# normally recovers the text layer nearly in full on a native page, so a page
# under 10% has lost its body, not trimmed a header.
NATIVE_FALLBACK_RATIO = 0.10

# Formats worth accepting here. Anything not listed is refused by name rather
# than attempted and silently returned empty — the failure mode this whole
# change exists to remove.
#
# .md is deliberately absent (#1711). The image does not install docling's
# `format-markdown` extra, so a Markdown input died on a missing `marko` import
# and came back as a 422 "processing limits" error, which read like a document
# problem. Core never sends Markdown here anyway: it reads text-like files
# itself (documents.cjs PLAIN_TEXT, #577). Refusing by name gives a 415.
SUPPORTED = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods",
    ".html", ".htm", ".epub", ".csv", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
}

_converter = None
_figure_converter = None

# #1726 (multimodal RAG): figures, returned only when the caller asks (X-Document-Figures: 1). Each is
# a PNG at most FIGURE_MAX_SIDE pixels on its longer side; smaller than FIGURE_MIN_SIDE on either side
# (bullets, rules, logos) is skipped. At most FIGURE_CAP per document and FIGURE_TOTAL_CAP base64
# characters in all, so a reply stays far inside the 32 MiB frame (the text alone can take ~12 MiB
# escaped). FIGURE_IMAGES_SCALE is Docling's render scale for PDF crops (1.0 = 72 dpi).
FIGURE_CAP = 64
FIGURE_TOTAL_CAP = 12 * 1024 * 1024
FIGURE_MAX_SIDE = 1024
FIGURE_MIN_SIDE = 64
FIGURE_IMAGES_SCALE = 2.0
# A PDF page whose own text layer has fewer visible characters than this, yet came back from Docling
# with text, was read by OCR: a scanned page. Its figures are never returned (core embeds such a page
# as text only) and the page says `"ocr": true`.
OCR_TEXT_LAYER_MIN = 20
# An image input is itself the image: it has no figures to return.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def _build_converter(figures=False):
    """One converter, reused. Construction loads the layout and table models.

    #1726: the figures variant also renders each picture's crop (generate_picture_images); it is a
    second converter, built the first time figures are asked for, so a request without them works
    exactly as before.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.table_structure_options.do_cell_matching = True
    # Tesseract, not Docling's EasyOCR default: EasyOCR measures ~13 s/page on
    # CPU and would dominate everything else. Tesseract is already in this
    # image and is what the previous OCR worker used, so output stays
    # comparable. Languages match services/ocr/server.py.
    options.ocr_options = TesseractCliOcrOptions(lang=["eng", "deu"])
    if figures:
        options.generate_picture_images = True
        options.images_scale = FIGURE_IMAGES_SCALE
    # Charts are deliberately NOT enabled. Chart extraction loads
    # granite-vision-3.3-2b-chart2csv, a 2B vision model, which is a different
    # order of cost from the ~1.6 GB layout+table pair and would contend with
    # llama.cpp. It is a separate decision, not a default.
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def _convert(path, figures=False, pages=None):
    """Convert, bounded to PAGE_CAP pages of actual work.

    `page_range` matters more than it looks. Without it Docling converts every
    page and `extract` then throws the overflow away, so a 334-page book costs
    full time and memory to produce 300 pages of output. Measured on DaServer,
    that is not merely wasteful: a 334-page PDF was OOM-killed (SIGKILL, exit
    137) against a 4 GB limit after 520 s. Bounding the conversion bounds peak
    RSS and wall time together.

    The true length is still reported, so `truncatedPages` stays honest: it is
    read from the input document rather than from the pages we chose to
    convert.
    """
    global _converter, _figure_converter
    if figures:
        if _figure_converter is None:
            _figure_converter = _build_converter(figures=True)
        converter = _figure_converter
    else:
        if _converter is None:
            _converter = _build_converter()
        converter = _converter
    # #1864: `pages` converts only that range (absolute page numbers), from a file up to RANGED_FILE_SIZE_CAP.
    limits = {"max_file_size": RANGED_FILE_SIZE_CAP if pages else FILE_SIZE_CAP}
    if not str(path).lower().endswith(".pdf"):
        limits["max_num_pages"] = NON_PDF_MAX_PAGES
    result = converter.convert(path, page_range=tuple(pages) if pages else (1, PAGE_CAP), **limits)
    source_pages = getattr(getattr(result, "input", None), "page_count", None)
    if not isinstance(source_pages, bool) and isinstance(source_pages, int) and source_pages > 0:
        return Converted(result.document, source_pages)
    return Converted(result.document, None)


def _pages_from(document):
    """Group the converted document's items into per-page text, in reading order.

    Docling has already decided reading order and table structure; this only
    flattens its result per page. Tables are emitted as Markdown so the
    structure survives into the chunker and into the model's context, which is
    the entire point of the change.
    """
    from docling_core.types.doc import TableItem, TextItem

    pages = {}
    for item, _level in document.iterate_items():
        page_no = None
        prov = getattr(item, "prov", None)
        if prov:
            page_no = getattr(prov[0], "page_no", None)
        if page_no is None:
            page_no = 1
        if isinstance(item, TableItem):
            piece = item.export_to_markdown(doc=document)
        elif isinstance(item, TextItem):
            piece = item.text
        else:
            continue
        if piece and piece.strip():
            pages.setdefault(int(page_no), []).append(piece.strip())
    return pages


def _pictures_from(document):
    """(page, image, caption, text) for each picture Docling found, in document order (#1726).

    `image` is a PIL image (None when Docling kept none), `caption` its caption text, `text` the text
    Docling read inside the picture region (chart labels, a diagram's words).
    """
    out = []
    for picture in getattr(document, "pictures", None) or []:
        prov = getattr(picture, "prov", None)
        page_no = getattr(prov[0], "page_no", None) if prov else None
        try:
            image = picture.get_image(document)
        except Exception:
            image = None
        try:
            caption = picture.caption_text(document) or ""
        except Exception:
            caption = ""
        words = []
        for ref in getattr(picture, "children", None) or []:
            try:
                item = ref.resolve(document)
            except Exception:
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                words.append(text.strip())
        out.append((int(page_no) if page_no else 1, image, caption, "\n".join(words)))
    return out


def _encode_png(image):
    """A PIL image as PNG bytes, at most FIGURE_MAX_SIDE on its longer side; None when too small."""
    import io

    width, height = getattr(image, "size", (0, 0))
    if width < FIGURE_MIN_SIDE or height < FIGURE_MIN_SIDE:
        return None
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    image = image.copy()
    image.thumbnail((FIGURE_MAX_SIDE, FIGURE_MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _native_page_text(path, numbers):
    """The PDF's own text layer for the given 1-based pages, via pypdfium2.

    pypdfium2 is already in the image: it is what docling's `format-pdf` extra
    uses for page rendering. It reads only the text layer — no layout, no OCR —
    which is exactly what makes it a safe witness: if it finds text, the text
    is there. Any failure (encrypted, malformed) returns what was read so far;
    the fallback is a safety net and must never fail a conversion that
    otherwise succeeded.
    """
    import pypdfium2 as pdfium

    texts = {}
    try:
        document = pdfium.PdfDocument(path)
    except Exception:
        return texts
    try:
        count = len(document)
        for number in numbers:
            if not 1 <= number <= count:
                continue
            try:
                page = document[number - 1]
                textpage = page.get_textpage()
                try:
                    texts[number] = textpage.get_text_range()
                finally:
                    textpage.close()
                    page.close()
            except Exception:
                continue
    finally:
        document.close()
    return texts


def _visible(text):
    return sum(1 for ch in text if not ch.isspace())


def _clean_native(text):
    """Normalise pdfium's text: CRLF line ends, trailing spaces, runs of blank lines."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out, gap = [], False
    for line in lines:
        if line:
            if gap and out:
                out.append("")
            out.append(line)
            gap = False
        else:
            gap = True
    return "\n".join(out).strip()


def extract(path, name, convert=_convert, pages_from=_pages_from, native_text=_native_page_text,
            figures=False, pictures_from=_pictures_from, encode_png=_encode_png, pages=None):
    """Convert one document. `convert`/`pages_from`/`native_text`/`pictures_from`/`encode_png` are
    injectable for tests. `figures` (#1726) also returns the document's figures and marks each page
    read by OCR (`"ocr": true`); without it the result is exactly what it always was. `pages` (#1864,
    PDFs only) is a (first, last) range: only those pages are converted and returned, `total` is still
    the whole document's page count, and `truncatedPages` says whether pages follow the range."""
    suffix = "." + str(name).rsplit(".", 1)[-1].lower() if "." in str(name) else ""
    if suffix not in SUPPORTED:
        raise ValueError("unsupported format")
    if pages and suffix != ".pdf":
        raise ValueError("page ranges are for PDFs only")

    kwargs = {**({"figures": True} if figures else {}), **({"pages": tuple(pages)} if pages else {})}
    converted = convert(path, **kwargs) if kwargs else convert(path)
    # Tolerant of both shapes: the real `_convert` returns a Converted, while a
    # test may inject a bare document.
    document = getattr(converted, "document", converted)
    source_pages = getattr(converted, "page_count", None)
    grouped = pages_from(document)
    # A page count from the document when it has one; otherwise infer from the
    # highest page we actually saw, so a single-page DOCX does not report 0.
    # Prefer the input's own page count: `document.pages` only holds the pages
    # that were actually converted, which _convert caps at PAGE_CAP.
    declared = source_pages
    if isinstance(declared, bool) or not isinstance(declared, int) or declared <= 0:
        converted_pages = getattr(document, "pages", {}) or {}
        # A range's converted pages are numbered absolutely (#1864): the highest is the most we know.
        declared = max((int(n) for n in converted_pages), default=0) if pages else len(converted_pages)
    total = declared or (max(grouped) if grouped else 1)

    first, last = (int(pages[0]), int(pages[1])) if pages else (1, PAGE_CAP)
    numbers = range(first, min(total, last) + 1)
    docling_text = {n: "\n\n".join(grouped.get(n, [])) for n in numbers}

    # Native-text fallback (#700), PDFs only. Why this rather than also walking
    # the text items Docling files under picture regions (iterate_items with
    # traverse_pictures=True): that would change the output of every page with
    # a figure — chart labels, axis ticks, logo text — for all documents, while
    # fixing only one of the ways a page can come back empty. This touches only
    # pages Docling left (near-)empty, works whatever the cause, and says so.
    fallback = {}
    if suffix == ".pdf":
        candidates = [n for n in numbers if _visible(docling_text[n]) < NATIVE_FALLBACK_MIN_CHARS]
        if candidates:
            try:
                native = native_text(path, candidates) or {}
            except Exception:
                # The safety net must not fail a conversion that succeeded;
                # without it the page is reported exactly as before (blank).
                native = {}
            for n, raw in native.items():
                text = _clean_native(str(raw or ""))
                seen = _visible(text)
                if n in docling_text and seen >= NATIVE_FALLBACK_MIN_CHARS \
                        and _visible(docling_text[n]) < seen * NATIVE_FALLBACK_RATIO:
                    fallback[n] = text

    results = []
    remaining = TOTAL_TEXT_CAP
    for number in numbers:
        text = fallback.get(number, docling_text[number])
        budget = max(0, min(PAGE_TEXT_CAP, remaining))
        truncated = len(text) > budget
        text = text[:budget]
        remaining -= len(text)
        reason = None
        if number in fallback:
            # Degraded rather than native: the words are right, but reading
            # order, columns and tables are pdfium's raw text order, not
            # Docling's layout. The caller shows the document as partial.
            status, reason = "degraded", "native-fallback"
        elif truncated:
            status = "truncated"
        elif text.strip():
            status = "native"
        else:
            # "blank" means Docling returned nothing for this page, which is
            # NOT the same as the page being empty. Measured on a scanned page
            # of handwriting: Tesseract cannot read handwriting, Docling's
            # layout model returned zero clusters, and the page arrived here
            # indistinguishable from a genuinely blank one. There is no signal
            # in the conversion result that separates the two cases, so this
            # deliberately does not guess. Callers must treat "blank" as "no
            # text recovered", not as "no content present" — see the known gap
            # in services/docling/README.md.
            status = "blank"
        page = {"number": number, "text": text, "status": status,
                "method": "pdfium" if reason else "docling", "truncated": truncated}
        if reason:
            page["reason"] = reason
        results.append(page)
    out = {"pages": results, "total": total, "truncatedPages": total > last}
    if figures:
        out["figures"] = _figures(path, suffix, document, results, docling_text, fallback, native_text, pictures_from, encode_png)
    return out


def _figures(path, suffix, document, results, docling_text, fallback, native_text, pictures_from, encode_png):
    """#1726: mark the pages read by OCR, then return the figures of every other page."""
    import base64

    scanned = set()
    if suffix in IMAGE_SUFFIXES:
        scanned = {p["number"] for p in results if p["text"].strip()}
    elif suffix == ".pdf":
        try:
            layer = native_text(path, [p["number"] for p in results]) or {}
        except Exception:
            layer = {}
        for p in results:
            n = p["number"]
            if n not in fallback and _visible(docling_text.get(n, "")) > 0 and _visible(str(layer.get(n, "") or "")) < OCR_TEXT_LAYER_MIN:
                scanned.add(n)
    for p in results:
        p["ocr"] = p["number"] in scanned
    if suffix in IMAGE_SUFFIXES:
        return []
    known = {p["number"] for p in results}
    found, spent = [], 0
    try:
        pictures = pictures_from(document)
    except Exception:
        pictures = []
    for index, (page, image, caption, text) in enumerate(pictures):
        if len(found) >= FIGURE_CAP:
            break
        if page not in known or page in scanned or image is None:
            continue
        try:
            data = encode_png(image)
        except Exception:
            data = None
        if not data:
            continue
        encoded = base64.b64encode(data).decode("ascii")
        if spent + len(encoded) > FIGURE_TOTAL_CAP:
            continue
        spent += len(encoded)
        figure = {"page": page, "index": index, "mime": "image/png", "data": encoded}
        if caption and caption.strip():
            figure["caption"] = caption.strip()[:300]
        if text and text.strip():
            figure["text"] = text.strip()[:4000]
        found.append(figure)
    return found
