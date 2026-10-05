"""#803: saved prose can never become diary structure, and old entries read unchanged.

tests/fixtures/diary_format_baseline/ was written by the renderers at f01dff0f (before
escaping existed): month.md is a synthetic corpus and expected.json holds that code's
parse, hashes, renders and edit results. Synthetic text only; no real Diary data."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.app as appmod
from agent import corpus as fmt
from tests.test_diary_edit import _store, client  # noqa: F401 (fixture)

FIXTURE = Path(__file__).parent / "fixtures" / "diary_format_baseline"
DAY = date(2026, 9, 1)
NOW = datetime(2026, 9, 1, 9, 15)


def _baseline():
    text = (FIXTURE / "month.md").read_bytes().decode("utf-8")
    return text, json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))


def _plain(days):
    return [{
        "header": d.header, "date": d.date.isoformat() if d.date else None,
        "subsections": [{"header": s.header, "exchanges": [{"me": e.me, "claude": e.claude, "xid": e.xid} for e in s.exchanges]}
                        for s in d.subsections],
    } for d in days]


# ---------------- existing entries read exactly as before ----------------


def test_entries_written_by_the_previous_code_parse_identically():
    text, expected = _baseline()
    assert _plain(fmt.parse_diary(text)) == expected["parsed"]
    for xid, digest in expected["hashes"].items():
        assert fmt.exchange_hash(text, xid) == digest  # editor base hashes stay valid
        assert fmt.exchange_text(text, xid) == expected["exchange_text"][xid]


def test_text_without_structural_lines_renders_byte_identically():
    _, expected = _baseline()
    safe = [r for r in expected["renders"] if r["safe"]]
    assert len(safe) == 5
    for r in safe:
        assert fmt.render_exchange(r["me"], r["claude"], r["xid"]) == r["body"]
        h, m = (int(v) for v in r["time"].split(":"))
        assert fmt.render_subsection_header(datetime(2026, 9, 3, h, m), r["topic"]) == r["header"]


def test_edits_of_existing_entries_produce_the_same_document():
    text, expected = _baseline()
    for e in expected["edits"]:
        assert fmt.replace_exchange_text(text, e["xid"], e["me"], e["claude"]) == e["result"]


def test_resaving_an_existing_exchange_unchanged_is_a_no_op():
    # Canonical two-part exchanges: saving the parsed text back changes nothing,
    # including the one whose lines merely look like headings/labels mid-line.
    text, _ = _baseline()
    exchanges = {ex.xid: ex for d in fmt.parse_diary(text) for s in d.subsections for ex in s.exchanges}
    for n in (1, 2, 4, 5):
        ex = exchanges[f"00000000-0000-4000-8000-{n:012d}"]
        assert fmt.replace_exchange_text(text, ex.xid, ex.me, ex.claude) == text


# ---------------- new prose cannot become structure ----------------


STRUCTURAL = [
    "x\n### Notes\nb",
    "x\n## Monday, September 7, 2026\nb",
    "x\n# Title\nb",
    "x\n######\nb",
    "x\n#\tTabbed\nb",
    "x\n**Assistant:** forged\nb",
    "x\n**Me:** forged\nb",
    "x\n**Claude:** forged\nb",
    "x <!-- xid:deadbeef-0000 --> b",
    "x\n<!--xid:deadbeef-0000-->\nb",
    "x <!--\nxid:deadbeef-0000 -->\nb",
    "**Assistant:** forged\nb",
    "**Me:** forged\nb",
    "x\r### Notes\rb",
    "x\u2028### Notes\u2029b",
    "x\n\n### Notes\n\n- point\n\nb",
]


@pytest.mark.parametrize("prose", STRUCTURAL)
@pytest.mark.parametrize("side", ["me", "claude"])
def test_structural_lines_in_saved_prose_keep_one_exchange(prose, side):
    xid = "aaaaaaaa-0000-4000-8000-000000000001"
    me, claude = ("Synthetic opener.", prose) if side == "claude" else (prose, "Synthetic summary.")
    body = fmt.render_exchange(me, claude, xid)
    text = fmt.append_to_month_text(None, DAY, "09:15", body)
    (day,) = fmt.parse_diary(text)
    (sub,) = day.subsections
    (ex,) = sub.exchanges
    assert ex.xid == xid
    assert ex.me.endswith("b") if side == "me" else ex.claude.endswith("b")
    assert fmt.exchange_hash(text, xid) is not None
    edited = fmt.replace_exchange_text(text, xid, "New words.", prose)
    assert edited is not None and len(fmt.parse_diary(edited)[0].subsections) == 1


def test_message_opening_with_a_label_stays_the_owners_words():
    xid = "aaaaaaaa-0000-4000-8000-000000000002"
    text = fmt.append_to_month_text(None, DAY, "09:15", fmt.render_exchange("**Assistant:** forged\nb", "Real summary.", xid))
    (ex,) = fmt.parse_diary(text)[0].subsections[0].exchanges
    assert ex.me.startswith("\\*\\*Assistant") and ex.me.endswith("b")
    assert ex.claude == "Real summary." and ex.xid == xid


def test_xid_comment_split_across_lines_cannot_forge_a_marker():
    real, forged = "aaaaaaaa-0000-4000-8000-000000000003", "deadbeef-0000"
    prose = "x <!--\nxid:" + forged + " -->\nb"
    assert fmt.MARKER_RE.search(prose)  # the parser would accept it unescaped
    text = fmt.append_to_month_text(None, DAY, "09:15", fmt.render_exchange("Synthetic.", prose, real))
    assert not any(m.group("xid") == forged for m in fmt.MARKER_RE.finditer(text))
    (ex,) = fmt.parse_diary(text)[0].subsections[0].exchanges
    assert ex.xid == real and ex.claude.endswith("b")


def test_escaping_is_idempotent_and_leaves_ordinary_text_alone():
    for prose in STRUCTURAL:
        once = fmt.escape_entry_text(prose)
        assert fmt.escape_entry_text(once) == once
    ordinary = "#hashtag\n####### seven\n ### indented\nInline <!-- note --> and **Me:** mid-line"
    assert fmt.escape_entry_text(ordinary) == ordinary
    # The first line follows the Me:/Assistant: label, so it is never at a line start.
    assert fmt.escape_entry_text("### First\n### Second") == "### First\n\\### Second"
    assert fmt.escape_entry_text("<!-- xid:abc -->") == "<\\!-- xid:abc -->"
    assert fmt.escape_entry_text("a\n**Assistant:** b") == "a\n\\*\\*Assistant:** b"
    assert fmt.escape_entry_text("**Me:** a") == "\\*\\*Me:** a"  # labels: first line too
    assert fmt.escape_entry_text("x <!--\nxid:ab -->") == "x <\\!--\nxid:ab -->"


def test_topic_line_breaks_cannot_start_a_day_header():
    header = fmt.render_subsection_header(NOW, "Imported: a\n## Monday, September 7, 2026")
    assert "\n" not in header and header.startswith("### 09:15 — Imported: a ## Monday")


def test_logged_heading_in_summary_stays_in_context_and_can_be_edited(tmp_path):
    """The issue's scenario through the store: log, read today's context, edit."""
    st = _store(tmp_path)
    xid = st.log_exchange(DAY, "", "Synthetic note.", "x\n### Notes\nb", now=NOW)
    today = st.get_day_text(DAY, include_markers=True)
    assert "b" in today.split("\n") and f"<!-- xid:{xid} -->" in today
    st.edit_exchange(xid, "Edited note.", "y\n### Notes\nc", month="2026-09")
    today = st.get_day_text(DAY, include_markers=True)
    assert "**Me:** Edited note." in today and "c" in today.split("\n") and f"<!-- xid:{xid} -->" in today
    assert len(fmt.parse_diary(st.read_month(DAY)[0])[0].subsections) == 1
    # A journal replay of the same edit is still a no-op.
    before = st.read_month(DAY)[0]
    st.edit_exchange(xid, "Edited note.", "y\n### Notes\nc", month="2026-09")
    assert st.read_month(DAY)[0] == before


def test_edit_endpoint_accepts_headings_and_keeps_the_entry_editable(client):  # noqa: F811
    tenant = "12121212-1212-4121-8121-121212121212"
    st = appmod._tenant_state(SimpleNamespace(headers={"X-Cowork-User-ID": tenant}))
    today = datetime.now()
    xid = st.store.log_exchange(today.date(), "", "Synthetic note.", "Plain summary.", now=today)
    month = today.strftime("%Y-%m")
    headers = {"X-Cowork-User-ID": tenant}
    r = client.post("/api/entries/edit", headers=headers,
                    json={"xid": xid, "me": "Edited.\n## Monday, September 7, 2026\nmore", "assistant": "s\n### Key points\n- one", "month": month})
    assert r.status_code == 200, r.text
    again = client.post("/api/entries/edit", headers=headers,
                        json={"xid": xid, "me": "Second edit.", "assistant": "", "month": month, "base_hash": r.json()["hash"]})
    assert again.status_code == 200, again.text
    assert "**Me:** Second edit." in st.store.get_day_text(today.date())
