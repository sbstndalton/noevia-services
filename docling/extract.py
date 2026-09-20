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
`status` is one of native | blank | truncated. The caller's vocabulary also
has ocr/unreadable/failed, which this extractor never emits: Docling runs OCR
inline rather than as a separate pass, and it cannot distinguish an
unreadable page from an empty one.
"""

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

# Formats worth accepting here. Anything not listed is refused by name rather
# than attempted and silently returned empty — the failure mode this whole
# change exists to remove.
SUPPORTED = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods",
    ".html", ".htm", ".md", ".epub", ".csv", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
}

_converter = None


def _build_converter():
    """One converter, reused. Construction loads the layout and table models."""
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
    # Charts are deliberately NOT enabled. Chart extraction loads
    # granite-vision-3.3-2b-chart2csv, a 2B vision model, which is a different
    # order of cost from the ~1.6 GB layout+table pair and would contend with
    # llama.cpp. It is a separate decision, not a default.
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def _convert(path):
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
    global _converter
    if _converter is None:
        _converter = _build_converter()
    result = _converter.convert(path, page_range=(1, PAGE_CAP))
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


def extract(path, name, convert=_convert, pages_from=_pages_from):
    """Convert one document. `convert`/`pages_from` are injectable for tests."""
    suffix = "." + str(name).rsplit(".", 1)[-1].lower() if "." in str(name) else ""
    if suffix not in SUPPORTED:
        raise ValueError("unsupported format")

    converted = convert(path)
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
        declared = len(getattr(document, "pages", {}) or {})
    total = declared or (max(grouped) if grouped else 1)

    results = []
    remaining = TOTAL_TEXT_CAP
    for number in range(1, min(total, PAGE_CAP) + 1):
        text = "\n\n".join(grouped.get(number, []))
        budget = max(0, min(PAGE_TEXT_CAP, remaining))
        truncated = len(text) > budget
        text = text[:budget]
        remaining -= len(text)
        if truncated:
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
        results.append({"number": number, "text": text, "status": status,
                        "method": "docling", "truncated": truncated})
    return {"pages": results, "total": total, "truncatedPages": total > PAGE_CAP}
