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
`status` is one of native | ocr | blank | unreadable | truncated | failed.
"""

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
    global _converter
    if _converter is None:
        _converter = _build_converter()
    return _converter.convert(path).document


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

    document = convert(path)
    grouped = pages_from(document)
    # A page count from the document when it has one; otherwise infer from the
    # highest page we actually saw, so a single-page DOCX does not report 0.
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
            # Docling ran OCR where it judged it necessary, so an empty page
            # here is empty, not un-attempted. Saying "blank" rather than
            # "ocr-needed" keeps the caller from queuing a second pass that
            # would find nothing.
            status = "blank"
        results.append({"number": number, "text": text, "status": status,
                        "method": "docling", "truncated": truncated})
    return {"pages": results, "total": total, "truncatedPages": total > PAGE_CAP}
