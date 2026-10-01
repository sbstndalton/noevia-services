"""Tests for the extraction contract, with Docling stubbed.

Docling itself is not exercised here — it needs 669 MB of models and the CI
container has neither them nor the network to fetch them. What IS testable is
everything around the conversion: format gating, page grouping, the caps, and
the status values the Node side branches on. selftest.py covers the rest, on
the host, against a real document.
"""
import extract as extractor
import pytest


class FakeProv:
    def __init__(self, page_no):
        self.page_no = page_no


class FakeText:
    def __init__(self, text, page):
        self.text = text
        self.prov = [FakeProv(page)]


class FakeTable:
    def __init__(self, markdown, page):
        self._markdown = markdown
        self.prov = [FakeProv(page)]

    def export_to_markdown(self, doc=None):
        return self._markdown


def fake_pages_from(items):
    """Stand in for _pages_from without importing docling_core's real types."""
    def grouped(_document):
        out = {}
        for item in items:
            piece = item.export_to_markdown() if isinstance(item, FakeTable) else item.text
            if piece and piece.strip():
                out.setdefault(item.prov[0].page_no, []).append(piece.strip())
        return out
    return grouped


def run(items, name="doc.pdf", total=None):
    document = type("Doc", (), {"pages": {n: None for n in range(1, (total or 1) + 1)}})()
    return extractor.extract("/unused", name, convert=lambda _p: document,
                             pages_from=fake_pages_from(items))


def test_unsupported_formats_are_refused_by_name_not_silently_empty():
    # The bug this replaces: .xlsx was accepted, filed as a Document, and then
    # produced empty content with no error anywhere.
    for name in ["archive.zip", "firmware.bin", "noextension"]:
        with pytest.raises(ValueError):
            run([], name=name)


def test_the_office_formats_that_previously_did_nothing_are_supported():
    for suffix in [".docx", ".pptx", ".xlsx", ".odt", ".odp", ".ods", ".epub"]:
        assert suffix in extractor.SUPPORTED, suffix


def test_items_are_grouped_by_page_and_joined_in_order():
    out = run([FakeText("first", 1), FakeText("second", 1), FakeText("later", 2)], total=2)
    assert [p["number"] for p in out["pages"]] == [1, 2]
    assert out["pages"][0]["text"] == "first\n\nsecond"
    assert out["pages"][1]["text"] == "later"
    assert all(p["method"] == "docling" for p in out["pages"])


def test_a_table_survives_as_markdown_so_its_structure_reaches_the_model():
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    out = run([FakeText("intro", 1), FakeTable(table, 1)])
    assert table in out["pages"][0]["text"]


def test_a_page_with_no_text_is_blank_not_ocr_needed():
    # Docling has already run OCR where it judged it necessary, so an empty
    # page is empty. Reporting 'ocr-needed' would queue a second pass that
    # finds nothing — which is what the old paint-op guess did.
    out = run([FakeText("only page one", 1)], total=2)
    assert out["pages"][0]["status"] == "native"
    assert out["pages"][1]["status"] == "blank"


def test_an_oversized_page_is_capped_and_says_so():
    out = run([FakeText("x" * (extractor.PAGE_TEXT_CAP + 500), 1)])
    page = out["pages"][0]
    assert page["truncated"] is True
    assert page["status"] == "truncated"
    assert len(page["text"]) == extractor.PAGE_TEXT_CAP


def test_the_whole_document_budget_is_shared_across_pages():
    per_page = extractor.PAGE_TEXT_CAP
    wanted = extractor.TOTAL_TEXT_CAP // per_page + 2
    out = run([FakeText("y" * per_page, n) for n in range(1, wanted + 1)], total=wanted)
    assert sum(len(p["text"]) for p in out["pages"]) <= extractor.TOTAL_TEXT_CAP
    assert out["pages"][-1]["text"] == "", "the budget runs out rather than being exceeded"


def test_page_count_is_capped_and_the_overflow_is_reported():
    out = run([FakeText("z", 1)], total=extractor.PAGE_CAP + 40)
    assert len(out["pages"]) == extractor.PAGE_CAP
    assert out["truncatedPages"] is True
    assert out["total"] == extractor.PAGE_CAP + 40, "the real length is still reported"


def test_a_document_declaring_no_pages_still_reports_one():
    document = type("Doc", (), {"pages": {}})()
    out = extractor.extract("/unused", "memo.docx", convert=lambda _p: document,
                            pages_from=fake_pages_from([FakeText("body", 1)]))
    assert out["total"] == 1
    assert out["pages"][0]["text"] == "body"


class StrictDoc:
    """A stand-in for DoclingDocument, which is a pydantic model.

    The point is the __setattr__: pydantic refuses attributes that are not
    declared fields. An earlier version of _convert stashed the page count on
    the document with setattr() and passed every test here, because the old
    stub was a plain object that accepted anything. On the real type it raised
    ValueError — after paying 454 s to convert a 334-page PDF. Stubs that are
    more permissive than the real thing are how that got through.
    """

    __slots__ = ("pages",)

    def __init__(self, pages):
        object.__setattr__(self, "pages", pages)

    def __setattr__(self, name, value):
        raise ValueError(f'"StrictDoc" object has no field "{name}"')


def test_the_converted_document_is_never_mutated():
    # Regression: _convert must not write to the document it returns.
    document = StrictDoc({1: None})
    out = extractor.extract("/unused", "memo.docx",
                            convert=lambda _p: extractor.Converted(document, 12),
                            pages_from=fake_pages_from([FakeText("body", 1)]))
    assert out["total"] == 12


def test_the_true_length_is_read_from_the_input_not_the_converted_pages():
    """_convert caps conversion at PAGE_CAP, so document.pages undercounts.

    Regression for a real measurement: a 334-page PDF is converted as 300
    pages, and reporting `total` from the converted set would claim the
    document is exactly 300 pages long and set truncatedPages False — quietly
    losing the fact that 34 pages were never looked at.
    """
    document = StrictDoc({n: None for n in range(1, extractor.PAGE_CAP + 1)})
    out = extractor.extract("/unused", "book.pdf",
                            convert=lambda _p: extractor.Converted(document, 334),
                            pages_from=fake_pages_from([FakeText("body", 1)]))
    assert out["total"] == 334, "the real length survives the conversion cap"
    assert out["truncatedPages"] is True
    assert len(out["pages"]) == extractor.PAGE_CAP


def test_a_page_count_that_is_absent_or_nonsense_falls_back_to_the_document():
    # Non-paginated formats and stubbed converters may not carry one. True is
    # included because bool is an int in Python and 1 page would be a lie.
    for bogus in [None, 0, -5, "many", True]:
        document = StrictDoc({1: None, 2: None})
        out = extractor.extract("/unused", "memo.docx",
                                convert=lambda _p: extractor.Converted(document, bogus),
                                pages_from=fake_pages_from([FakeText("body", 1)]))
        assert out["total"] == 2, f"fell back for {bogus!r}"
        assert out["truncatedPages"] is False


def test_a_bare_document_without_a_page_count_still_works():
    # The seam stays tolerant of a plain document, which is what most of the
    # tests above inject.
    out = run([FakeText("body", 1)], total=3)
    assert out["total"] == 3


def test_conversion_is_bounded_to_page_cap_and_the_document_is_not_touched():
    """The cap must bound the work, not just the output.

    Measured: without page_range a 334-page PDF was OOM-killed at a 4 GB limit
    after 520 s (exit 137). This pins the page_range argument so the bound is
    not silently dropped in a refactor, and uses StrictDoc so a future
    setattr() on the document fails here rather than in production.
    """
    seen = {}
    document = StrictDoc({1: None})

    class FakeResult:
        def __init__(self):
            self.document = document
            self.input = type("In", (), {"page_count": 334})()

    class FakeConverter:
        def convert(self, path, page_range=None):
            seen["page_range"] = page_range
            return FakeResult()

    extractor._converter = FakeConverter()
    try:
        converted = extractor._convert("/unused")
    finally:
        extractor._converter = None
    assert seen["page_range"] == (1, extractor.PAGE_CAP)
    assert converted.document is document
    assert converted.page_count == 334


# ── #700: a page with real text is never silently reported blank ──────────
#
# The Docling side is still stubbed (no models in CI), but the native text
# layer is read for real, by pypdfium2, from synthetic PDFs built at test time.
import synthetic_pdfs  # noqa: E402


def fixture(tmp_path, pages, name="doc.pdf"):
    return str(synthetic_pdfs.write(tmp_path / name, pages))


def run_pdf(path, items, total, name="doc.pdf", native_text=extractor._native_page_text):
    document = type("Doc", (), {"pages": {n: None for n in range(1, total + 1)}})()
    return extractor.extract(path, name, convert=lambda _p: document,
                             pages_from=fake_pages_from(items), native_text=native_text)


def test_text_inside_a_picture_region_falls_back_to_the_native_text_layer(tmp_path):
    # The #700 shape: a full-page image with selectable text on top. Docling's
    # layout can file the whole page as one picture and skip its text; the
    # stub returns nothing for the page, as Docling did on the real document.
    lines = synthetic_pdfs.page_lines()
    path = fixture(tmp_path, [("picture-text", lines)])
    page = run_pdf(path, [], total=1)["pages"][0]
    assert page["status"] == "degraded"
    assert page["reason"] == "native-fallback"
    assert page["method"] == "pdfium"
    assert lines[0] in page["text"] and lines[-1] in page["text"]


def test_a_near_empty_docling_page_also_falls_back(tmp_path):
    # Docling kept a caption and lost the body: still a silent loss.
    path = fixture(tmp_path, [("picture-text", synthetic_pdfs.page_lines())])
    page = run_pdf(path, [FakeText("Figure 1", 1)], total=1)["pages"][0]
    assert page["status"] == "degraded"
    assert "aisle 12" in page["text"]


def test_a_text_page_docling_read_is_left_alone_and_pdfium_is_not_consulted(tmp_path):
    lines = synthetic_pdfs.page_lines()
    path = fixture(tmp_path, [("text", lines)])
    calls = []

    def spy(p, numbers):
        calls.append(numbers)
        return extractor._native_page_text(p, numbers)

    page = run_pdf(path, [FakeText("\n".join(lines), 1)], total=1, native_text=spy)["pages"][0]
    assert page["status"] == "native"
    assert page["method"] == "docling"
    assert "reason" not in page
    assert calls == [], "a page Docling read in full costs no second parse"


def test_a_truly_blank_page_stays_blank(tmp_path):
    path = fixture(tmp_path, [("blank", [])])
    page = run_pdf(path, [], total=1)["pages"][0]
    assert page["status"] == "blank"
    assert page["text"] == ""


def test_a_picture_with_no_text_layer_stays_blank(tmp_path):
    # A scan with no text layer has nothing to fall back to; it is reported
    # exactly as before rather than invented.
    path = fixture(tmp_path, [("picture", [])])
    assert run_pdf(path, [], total=1)["pages"][0]["status"] == "blank"


def test_a_short_text_layer_below_the_threshold_is_not_promoted(tmp_path):
    # A running head or page number in the text layer is not "substantial".
    path = fixture(tmp_path, [("picture-text", ["Page 3 of 9"])])
    assert run_pdf(path, [], total=1)["pages"][0]["status"] == "blank"


def test_the_threshold_is_configurable(tmp_path, monkeypatch):
    path = fixture(tmp_path, [("picture-text", ["Page 3 of 9"])])
    monkeypatch.setattr(extractor, "NATIVE_FALLBACK_MIN_CHARS", 5)
    assert run_pdf(path, [], total=1)["pages"][0]["status"] == "degraded"
    monkeypatch.setenv("DOCLING_NATIVE_FALLBACK_MIN_CHARS", "350")
    assert extractor._env_int("DOCLING_NATIVE_FALLBACK_MIN_CHARS", 200) == 350
    for bogus in ["", "lots", "0", "-4"]:
        monkeypatch.setenv("DOCLING_NATIVE_FALLBACK_MIN_CHARS", bogus)
        assert extractor._env_int("DOCLING_NATIVE_FALLBACK_MIN_CHARS", 200) == 200, bogus


def test_a_mixed_document_reports_each_page_for_what_it_is(tmp_path):
    lines = synthetic_pdfs.page_lines()
    path = fixture(tmp_path, [("text", lines), ("picture-text", synthetic_pdfs.page_lines(start=20)), ("blank", [])])
    out = run_pdf(path, [FakeText("\n".join(lines), 1)], total=3)
    assert [p["status"] for p in out["pages"]] == ["native", "degraded", "blank"]
    assert "aisle 20" in out["pages"][1]["text"]


def test_only_the_pages_docling_left_short_are_read_natively(tmp_path):
    path = fixture(tmp_path, [("text", synthetic_pdfs.page_lines()), ("blank", [])])
    seen = []
    run_pdf(path, [FakeText("x" * 500, 1)], total=2,
            native_text=lambda p, numbers: seen.append(list(numbers)) or {})
    assert seen == [[2]]


def test_a_failing_text_layer_read_never_fails_the_conversion(tmp_path):
    def broken(_path, _numbers):
        raise RuntimeError("pdfium exploded")
    out = run_pdf("/unused", [], total=1, native_text=broken)
    assert out["pages"][0]["status"] == "blank"


def test_an_unreadable_file_yields_no_native_text():
    assert extractor._native_page_text("/does/not/exist.pdf", [1]) == {}


def test_non_pdf_formats_never_consult_the_pdf_text_layer():
    def forbidden(_path, _numbers):
        raise AssertionError("pdfium must not be asked about a DOCX")
    document = type("Doc", (), {"pages": {1: None}})()
    out = extractor.extract("/unused", "memo.docx", convert=lambda _p: document,
                            pages_from=fake_pages_from([]), native_text=forbidden)
    assert out["pages"][0]["status"] == "blank"


def test_fallback_text_obeys_the_page_cap_and_stays_degraded(tmp_path, monkeypatch):
    path = fixture(tmp_path, [("picture-text", synthetic_pdfs.page_lines())])
    monkeypatch.setattr(extractor, "PAGE_TEXT_CAP", 300)
    page = run_pdf(path, [], total=1)["pages"][0]
    assert page["status"] == "degraded"
    assert page["truncated"] is True
    assert len(page["text"]) == 300
