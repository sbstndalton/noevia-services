"""Tests for the extraction contract, with Docling stubbed.

Docling itself is not exercised here — it needs ~1.6 GB of models and the CI
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
