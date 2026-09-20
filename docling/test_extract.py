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
