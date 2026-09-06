"""External diary-like source detection and manual import (Phase 4).

Covers the read-only scanner (env parsing, date inference, safety caps, path
traversal) and the two endpoints (scan listing, explicit one-file import that
writes a dated corpus entry without touching the original file).
"""
from __future__ import annotations

import os
from datetime import datetime

import pytest

import agent.app as appmod
from agent.external_sources import (
    external_source_paths,
    infer_date,
    resolve_import_target,
    scan_source,
)
from tests.test_pipeline import client  # noqa: F401 — app-level TestClient fixture


# ---------------- env parsing ----------------


def test_external_source_paths_parses_and_dedupes(monkeypatch):
    monkeypatch.setenv(
        "DIARY_EXTERNAL_SOURCES",
        f"/data/journal, /data/journal/ ,/data/exports,/data/journal",
    )
    paths = external_source_paths()
    assert paths == ["/data/journal", "/data/exports"]  # trimmed, deduped, order kept


def test_external_source_paths_empty_when_unset(monkeypatch):
    monkeypatch.delenv("DIARY_EXTERNAL_SOURCES", raising=False)
    assert external_source_paths() == []


# ---------------- date inference ----------------


def test_infer_date_prefers_filename_iso_pattern(tmp_path):
    f = tmp_path / "2026-09-06-notes.md"
    f.write_text("x", encoding="utf-8")
    date, source = infer_date(f)
    assert (date.isoformat(), source) == ("2026-09-06", "filename")


def test_infer_date_handles_compact_pattern(tmp_path):
    f = tmp_path / "journal-20260906.txt"
    f.write_text("x", encoding="utf-8")
    date, source = infer_date(f)
    assert (date.isoformat(), source) == ("2026-09-06", "filename")


def test_infer_date_rejects_impossible_calendar_dates(tmp_path):
    f = tmp_path / "20261345.md"  # matches the compact shape, not a real date
    f.write_text("x", encoding="utf-8")
    date, source = infer_date(f)
    assert source == "mtime"
    assert date is not None


def test_infer_date_falls_back_to_mtime(tmp_path):
    f = tmp_path / "no-date-here.txt"
    f.write_text("x", encoding="utf-8")
    date, source = infer_date(f)
    assert source == "mtime"
    expected = datetime.fromtimestamp(f.stat().st_mtime).date().isoformat()
    assert date.isoformat() == expected


# ---------------- scanning ----------------


def test_scan_source_lists_only_allowed_extensions_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "2026-09-01.md").write_text("a", encoding="utf-8")
    (tmp_path / "sub" / "note.txt").write_text("b", encoding="utf-8")
    (tmp_path / "sub" / "extra.markdown").write_text("c", encoding="utf-8")
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")
    result = scan_source(str(tmp_path))
    names = {f["name"] for f in result["files"]}
    assert names == {"2026-09-01.md", "note.txt", "extra.markdown"}
    assert result["total"] == 3
    assert result["exists"] is True and result["truncated"] is False


def test_scan_source_reports_missing_path_without_raising(tmp_path):
    result = scan_source(str(tmp_path / "not-mounted-yet"))
    assert result["exists"] is False
    assert result["files"] == [] and result["total"] == 0


def test_scan_source_caps_file_count(tmp_path, monkeypatch):
    from agent import external_sources as es

    monkeypatch.setattr(es, "MAX_FILES_PER_SOURCE", 3)
    for i in range(6):
        (tmp_path / f"2026-09-0{i}.md").write_text("x", encoding="utf-8")
    result = scan_source(str(tmp_path))
    assert result["total"] == 3
    assert result["truncated"] is True


# ---------------- import target resolution ----------------


def test_resolve_import_target_rejects_traversal_and_absolute(tmp_path):
    assert resolve_import_target(str(tmp_path), "../escape.txt") is None
    assert resolve_import_target(str(tmp_path), "/etc/passwd") is None
    assert resolve_import_target(str(tmp_path), "") is None


def test_resolve_import_target_accepts_real_file(tmp_path):
    f = tmp_path / "2026-09-06.md"
    f.write_text("hello", encoding="utf-8")
    resolved = resolve_import_target(str(tmp_path), "2026-09-06.md")
    assert resolved is not None and resolved.name == "2026-09-06.md"


def test_resolve_import_target_rejects_disallowed_extension(tmp_path):
    f = tmp_path / "2026-09-06.exe"
    f.write_text("x", encoding="utf-8")
    assert resolve_import_target(str(tmp_path), "2026-09-06.exe") is None


# ---------------- endpoints ----------------


@pytest.fixture
def ext_source(tmp_path):
    src = tmp_path / "old-journal"
    src.mkdir()
    (src / "2026-08-15.md").write_text(
        "A quiet day. Walked the long way home and finally called my brother.\n",
        encoding="utf-8",
    )
    return src


def test_scan_endpoint_disabled_when_no_paths(client, monkeypatch):
    monkeypatch.delenv("DIARY_EXTERNAL_SOURCES", raising=False)
    r = client.get("/api/external-sources")
    assert r.status_code == 200
    assert r.json() == {"configured": False, "sources": [], "total": 0}


def test_scan_endpoint_reports_detected_files(client, monkeypatch, ext_source):
    monkeypatch.setenv("DIARY_EXTERNAL_SOURCES", str(ext_source))
    r = client.get("/api/external-sources")
    body = r.json()
    assert r.status_code == 200 and body["configured"] is True and body["total"] == 1
    f = body["sources"][0]["files"][0]
    assert f["name"] == "2026-08-15.md"
    assert f["date"] == "2026-08-15" and f["date_source"] == "filename"


def test_import_endpoint_writes_dated_entry_and_keeps_original(
    client, monkeypatch, ext_source
):
    monkeypatch.setenv("DIARY_EXTERNAL_SOURCES", str(ext_source))
    original = (ext_source / "2026-08-15.md").read_text(encoding="utf-8")

    r = client.post(
        "/api/external-sources/import",
        json={"source_path": str(ext_source), "rel_path": "2026-08-15.md"},
    )
    body = r.json()
    assert r.status_code == 200 and body["imported"] is True
    assert body["day"] == "2026-08-15" and body["date_source"] == "filename"

    # The entry landed in the corpus on the file's own date, verbatim.
    from datetime import date as _date

    text, _ = appmod.get_state().store.read_month(_date(2026, 8, 15))
    assert "A quiet day. Walked the long way home" in text
    assert "Imported: 2026-08-15.md" in text

    # The original file is byte-for-byte untouched.
    assert (ext_source / "2026-08-15.md").read_text(encoding="utf-8") == original


def test_import_endpoint_rejects_unconfigured_source(client, monkeypatch, ext_source):
    monkeypatch.setenv("DIARY_EXTERNAL_SOURCES", str(ext_source))
    r = client.post(
        "/api/external-sources/import",
        json={"source_path": "/somewhere/else", "rel_path": "2026-08-15.md"},
    )
    assert r.status_code == 400


def test_import_endpoint_rejects_traversal(client, monkeypatch, ext_source):
    monkeypatch.setenv("DIARY_EXTERNAL_SOURCES", str(ext_source))
    r = client.post(
        "/api/external-sources/import",
        json={"source_path": str(ext_source), "rel_path": "../../secrets.txt"},
    )
    assert r.status_code == 404
