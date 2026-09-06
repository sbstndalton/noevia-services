from datetime import date

from agent import corpus as fmt


SAMPLE = """# Diary

Some preamble that should be ignored.

## Saturday, August 29, 2026

### 09:15 — Morning

**Me:** Rough night, couldn't sleep past 4am.

**Claude:** The user described a rough night; the companion asked about caffeine timing.

<!-- xid:11111111-1111-1111-1111-111111111111 -->

### 20:02 — Evening walk

**Me:** Walk helped a lot honestly.

**Claude:** The user found the evening walk restorative; the companion noted the pattern.

<!-- xid:22222222-2222-2222-2222-222222222222 -->

## Sunday, August 30, 2026

### 08:00 — Today

**Me:** Up early, feeling better.

**Claude:** Improvement noted versus the prior two days.

<!-- xid:33333333-3333-3333-3333-333333333333 -->
"""


def test_parse_diary_extracts_days_subsections_exchanges():
    days = fmt.parse_diary(SAMPLE)
    assert len(days) == 2
    assert days[0].date == date(2026, 8, 29)
    assert [s.header for s in days[0].subsections] == ["09:15 — Morning", "20:02 — Evening walk"]
    ex = days[0].subsections[0].exchanges[0]
    assert ex.me == "Rough night, couldn't sleep past 4am."
    assert ex.xid == "11111111-1111-1111-1111-111111111111"
    assert "xid" not in ex.claude  # markers stripped from visible text


def test_parse_diary_ignores_preamble_and_stray_text():
    days = fmt.parse_diary(SAMPLE)
    # stray preamble produced no day sections and no phantom content
    assert days[0].header == "## Saturday, August 29, 2026"


def test_append_creates_new_month_file():
    out = fmt.append_to_month_text(None, date(2026, 9, 1), "10:00 — Start", "**Me:** hi\n\n**Claude:** The user said hi.\n")
    assert out.startswith("## Tuesday, September 1, 2026")
    assert "### 10:00 — Start" in out
    assert "**Me:** hi" in out


def test_append_into_existing_day_and_new_day():
    day = date(2026, 8, 30)
    # existing day header -> appended inside that day, after its existing exchanges
    out = fmt.append_to_month_text(SAMPLE, day, "12:00 — Lunch", "BODY")
    assert "### 12:00 — Lunch" in out
    assert out.index("## Sunday, August 30, 2026") < out.index("### 12:00 — Lunch")
    assert out.index("xid:33333333") < out.index("### 12:00 — Lunch")  # after Sunday's earlier exchange
    assert out.rstrip().endswith("BODY")
    assert "## Monday" not in out  # no phantom day header
    # new day -> new header appended after existing content
    out2 = fmt.append_to_month_text(SAMPLE, date(2026, 8, 31), "09:00 — New day", "BODY2")
    assert "## Monday, August 31, 2026" in out2
    assert out2.index("## Sunday, August 30, 2026") < out2.index("## Monday, August 31, 2026")


def test_marker_helpers():
    xid = "abcd"
    text = fmt.render_exchange("me text", "claude text", xid)
    assert fmt.has_marker(text, xid)
    assert not fmt.has_marker(text, "other")
    assert "xid" not in fmt.strip_markers(text)


def test_index_roundtrip_and_edits():
    idx = fmt.parse_index(None)
    assert fmt.register_month(idx, "August 2026", "2026-08.md")
    assert not fmt.register_month(idx, "August 2026", "2026-08.md")  # idempotent
    changed = fmt.apply_index_edits(
        idx,
        open_question_ops=[{"action": "add", "text": "Should I switch jobs?"}],
        timeline_ops=[{"action": "add", "date": "2026-09-01", "text": "Started diary agent"}],
        today="2026-09-01",
    )
    assert changed
    rendered = idx.render()
    assert "- [ ] Should I switch jobs?" in rendered
    assert "- **2026-09-01** — Started diary agent" in rendered
    assert "[August 2026](2026-08.md)" in rendered

    # re-parse rendered output — standing sections survive roundtrip
    idx2 = fmt.parse_index(rendered)
    assert idx2.sections["Open Questions"] == ["- [ ] Should I switch jobs?"]
    assert idx2.month_links == ["- [August 2026](2026-08.md)"]


def test_resolve_open_question():
    idx = fmt.parse_index("- [ ] Should I switch jobs?\n")
    idx.sections["Open Questions"] = ["- [ ] Should I switch jobs?"]
    fmt.apply_index_edits(idx, open_question_ops=[{"action": "resolve", "text": "switch jobs"}], today="2026-09-01")
    assert idx.sections["Open Questions"] == ["- [x] Should I switch jobs?"]


def test_resolve_prefers_exact_bullet_match():
    # One bullet's text is a substring of another's: an exact echo must resolve
    # only its own bullet (loose substring matching alone would hit both).
    idx = fmt.parse_index("")
    idx.sections["Open Questions"] = [
        "- [ ] Should I switch jobs?",
        "- [ ] Should I switch jobs? (asking for a friend)",
    ]
    fmt.apply_index_edits(
        idx,
        open_question_ops=[{"action": "resolve", "text": "Should I switch jobs?"}],
        today="2026-09-03",
    )
    oq = idx.sections["Open Questions"]
    assert oq[0].startswith("- [x] ")
    assert oq[1].startswith("- [ ] ")


def test_resolve_handles_noncanonical_checkbox_spacing():
    """INDEX.md hand-edited in Nextcloud can carry '-[ ]', '- [  ]', or tab
    spacing. Resolve must still rewrite the bullet (leniently, like the match
    helper) and must report changed=True — the old exact '- [ ] ' mutation
    regex silently did nothing here."""
    for spacing, resolved in [
        ("-[ ] Should I switch jobs?", "-[x] Should I switch jobs?"),
        ("- [  ] Should I switch jobs?", "- [x] Should I switch jobs?"),
        ("-\t[ ] Should I switch jobs?", "-\t[x] Should I switch jobs?"),
    ]:
        idx = fmt.parse_index("")
        idx.sections["Open Questions"] = [spacing]
        changed = fmt.apply_index_edits(
            idx,
            open_question_ops=[{"action": "resolve", "text": "switch jobs"}],
            today="2026-09-04",
        )
        assert changed, f"resolve must count as a change for {spacing!r}"
        assert idx.sections["Open Questions"] == [resolved]


def test_resolve_of_already_resolved_bullet_reports_no_change():
    idx = fmt.parse_index("")
    idx.sections["Open Questions"] = ["- [x] Should I switch jobs?"]
    changed = fmt.apply_index_edits(
        idx,
        open_question_ops=[{"action": "resolve", "text": "switch jobs"}],
        today="2026-09-04",
    )
    assert not changed
