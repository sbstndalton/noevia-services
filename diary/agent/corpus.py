"""Diary format engine — parsing, rendering, and appending in the canonical diary format.

Canonical format (the user's spec, enforced mechanically — never LLM-generated structure):

    ## Saturday, August 30, 2026

    ### 14:32 — Getting things in order

    **Me:** <user's message, kept near-verbatim, unparaphrased>

    **Claude:** <assistant's reply condensed into third-person natural prose>

Hidden exchange markers (`<!-- xid:... -->`) are appended after each exchange block so the
applier can dedupe idempotently; they are stripped when the log is shown to the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

DAY_HEADER_RE = re.compile(r"^## (?:(?P<dow>[A-Za-z]+), )?(?P<month>[A-Za-z]+) (?P<day>\d{1,2}), (?P<year>\d{4})\s*$")
SUB_HEADER_RE = re.compile(r"^### (?P<rest>.+?)\s*$")
ME_RE = re.compile(r"^\*\*Me:\*\*[ \t]?", re.M)
CLAUDE_RE = re.compile(r"^\*\*Claude:\*\*[ \t]?", re.M)
MARKER_RE = re.compile(r"<!--\s*xid:(?P<xid>[0-9a-fA-F-]+)\s*-->")
TIME_PREFIX_RE = re.compile(r"^(?P<time>\d{1,2}:\d{2})(?:\s*[—–-]\s*(?P<topic>.*))?$")


@dataclass
class Exchange:
    """One Me:/Claude: pair inside a ### subsection."""

    me: str
    claude: str
    xid: Optional[str] = None


@dataclass
class SubSection:
    """A `### <time/topic>` chunk — the unit of retrieval."""

    header: str  # raw header text after '### ' (e.g. '14:32 — Getting things in order')
    exchanges: List[Exchange] = field(default_factory=list)

    @property
    def time(self) -> Optional[str]:
        m = TIME_PREFIX_RE.match(self.header)
        return m.group("time") if m else None

    @property
    def topic(self) -> str:
        m = TIME_PREFIX_RE.match(self.header)
        return (m.group("topic") or "").strip() if m else self.header.strip()


@dataclass
class DaySection:
    """A `## <Day>, <Month> <date>, <year>` day and its subsections."""

    header: str  # canonical header line
    date: Optional[date]
    subsections: List[SubSection] = field(default_factory=list)


# ---------------- rendering ----------------


def canonical_day_header(d: date) -> str:
    return f"## {d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year}"


def render_subsection_header(now: Optional[datetime] = None, topic: str = "") -> str:
    stamp = (now or datetime.now()).strftime("%H:%M")
    return f"### {stamp} — {topic.strip()}" if topic.strip() else f"### {stamp}"


def render_exchange(me: str, claude: str, xid: str) -> str:
    me_clean = me.strip()
    claude_clean = claude.strip()
    return (
        f"**Me:** {me_clean}\n\n"
        f"**Claude:** {claude_clean}\n\n"
        f"<!-- xid:{xid} -->\n"
    )


def render_subsection_block(header: str, body: str) -> str:
    return f"### {header.strip()}\n\n{body.strip()}\n"


# ---------------- parsing ----------------


def parse_diary(text: str) -> List[DaySection]:
    """Parse a monthly file into day sections and subsections.

    Tolerant of malformed input: anything before the first day header is ignored;
    stray text between subsections is attached to the current subsection's last
    exchange if one exists, else dropped.
    """
    days: List[DaySection] = []
    current_day: Optional[DaySection] = None
    current_sub: Optional[SubSection] = None
    buf: List[str] = []

    def flush_buf_into_sub() -> None:
        nonlocal buf
        if current_sub is not None and buf:
            _append_buf_to_sub(current_sub, "\n".join(buf).strip())
        buf = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        day_m = DAY_HEADER_RE.match(line)
        if day_m:
            flush_buf_into_sub()
            d = _parse_day_date(day_m)
            current_day = DaySection(header=line, date=d)
            days.append(current_day)
            current_sub = None
            continue
        sub_m = SUB_HEADER_RE.match(line)
        if sub_m and current_day is not None:
            flush_buf_into_sub()
            current_sub = SubSection(header=sub_m.group("rest").strip())
            current_day.subsections.append(current_sub)
            continue
        if current_day is not None:
            buf.append(raw_line)
    flush_buf_into_sub()
    return days


def _parse_day_date(m: re.Match) -> Optional[date]:
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    try:
        return date(int(m.group("year")), months[m.group("month").lower()], int(m.group("day")))
    except (KeyError, ValueError):
        return None


def _append_buf_to_sub(sub: SubSection, body: str) -> None:
    """Split a subsection body into Me:/Claude: exchanges, tolerating intermixed text.

    Multi-line messages are preserved verbatim (minus surrounding blank lines).
    """
    # Exchange boundaries: each '**Me:**' at line start starts a new exchange.
    parts = re.split(r"(?m)^(?=\*\*Me:\*\*)", body)
    for part in parts:
        part = part.strip("\n")
        if not part:
            continue
        me_m = ME_RE.search(part)
        if not me_m:
            # stray text without a Me: opener — attach to last exchange's claude side
            if sub.exchanges:
                sub.exchanges[-1].claude += "\n" + part
            continue
        if me_m.start() > 0 and sub.exchanges:
            # stray prefix before the Me: opener — attach to previous exchange
            sub.exchanges[-1].claude += "\n" + part[: me_m.start()].strip()
            part = part[me_m.start():]
            me_m = ME_RE.search(part)
        rest = part[me_m.end():]
        claude_m = CLAUDE_RE.search(rest)
        if claude_m:
            me_text = rest[: claude_m.start()].strip()
            claude_text = rest[claude_m.end():].strip()
        else:
            me_text = rest.strip()
            claude_text = ""
        xid_m = MARKER_RE.search(claude_text) or MARKER_RE.search(me_text)
        # remove markers from visible text; keep xid on the exchange
        xid = xid_m.group("xid") if xid_m else None
        claude_text = MARKER_RE.sub("", claude_text).strip()
        me_text = MARKER_RE.sub("", me_text).strip()
        sub.exchanges.append(Exchange(me=me_text, claude=claude_text, xid=xid))


# ---------------- markers ----------------


def has_marker(text: str, xid: str) -> bool:
    return f"<!-- xid:{xid} -->" in text


def strip_markers(text: str) -> str:
    return MARKER_RE.sub("", text)


# ---------------- appends ----------------


def append_to_month_text(month_text: Optional[str], day: date, sub_header: str, body: str) -> str:
    """Append a subsection block to a month file's text.

    - New day -> day header appended at the end of the file.
    - Existing day -> block inserted at the end of that day's section (before the next
      day header), so appends land inside the correct day even if the file's last day
      is not the target day.

    Duplicate-exchange protection is handled by the journal + marker scan before this.
    """
    header = canonical_day_header(day)
    block_lines = render_subsection_block(sub_header, body).rstrip("\n").split("\n")
    if month_text is None or not month_text.strip():
        return header + "\n\n" + "\n".join(block_lines) + "\n"

    lines = month_text.rstrip("\n").split("\n")

    def _strip_trailing_blanks(ls):
        while ls and ls[-1].strip() == "":
            ls.pop()
        return ls

    if header in month_text:
        starts = [i for i, l in enumerate(lines) if l.strip() == header.strip()]
        if not starts:
            # Header line exists as text but was mangled (e.g., text prepended to the line by
            # a concurrent editor) — fall back to appending at the end with a fresh header.
            pass
        else:
            start = starts[-1]
            end = len(lines)
            for j in range(start + 1, len(lines)):
                if DAY_HEADER_RE.match(lines[j]):
                    end = j
                    break
            new_lines = _strip_trailing_blanks(lines[:end])
            new_lines.append("")
            new_lines.extend(block_lines)
            new_lines.extend(lines[end:])
            return "\n".join(new_lines).rstrip("\n") + "\n"

    new_lines = _strip_trailing_blanks(lines[:])
    new_lines.extend(["", header, ""])
    new_lines.extend(block_lines)
    return "\n".join(new_lines).rstrip("\n") + "\n"


# ---------------- index (standing sections) ----------------

MONTH_LINK_RE = re.compile(r"^\s*-\s*\[(?P<label>[^\]]+)\]\((?P<file>[^)]+)\.md\)\s*$")
SECTION_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")


@dataclass
class IndexFile:
    """INDEX.md: month links + standing sections (## Open Questions, ## Timeline of Key Events)."""

    month_links: List[str] = field(default_factory=list)  # raw '- [label](file.md)' lines
    sections: "dict[str, List[str]]" = field(default_factory=dict)  # title -> bullet lines
    preamble: List[str] = field(default_factory=list)

    def render(self) -> str:
        out: List[str] = []
        out.extend(self.preamble)
        out.append("## Months")
        out.append("")
        out.extend(self.month_links)
        out.append("")
        for title, bullets in self.sections.items():
            out.append(f"## {title}")
            out.append("")
            if bullets:
                out.extend(bullets)
            else:
                out.append("_None yet._")
            out.append("")
        return "\n".join(out).strip() + "\n"


def parse_index(text: Optional[str]) -> IndexFile:
    idx = IndexFile(
        sections={"Open Questions": [], "Timeline of Key Events": []},
    )
    if not text:
        return idx
    current_section: Optional[str] = None
    in_months = False
    preamble_done = False
    for line in text.splitlines():
        if SECTION_RE.match(line):
            title = SECTION_RE.match(line).group("title").strip()  # type: ignore[union-attr]
            if title.lower() == "months":
                in_months = True
                current_section = None
                preamble_done = True
                continue
            in_months = False
            current_section = title
            idx.sections.setdefault(title, [])
            preamble_done = True
            continue
        if not preamble_done:
            idx.preamble.append(line)
            continue
        if in_months and MONTH_LINK_RE.match(line):
            idx.month_links.append(line.rstrip())
            continue
        if current_section is not None and line.strip().startswith(("-", "*")) and not line.strip().startswith("<!--"):
            idx.sections[current_section].append(line.rstrip())
        elif current_section is not None and line.strip().startswith("<!--"):
            idx.sections[current_section].append(line.rstrip())
    return idx


def register_month(idx: IndexFile, label: str, filename: str) -> bool:
    """Add a month link if missing. Returns True if changed."""
    link = f"- [{label}]({filename})"
    if any(link in existing for existing in idx.month_links):
        return False
    idx.month_links.append(link)
    return True


def _bullet_match_indices(bullets: List[str], text: str) -> set:
    """Indices of bullets an index-edit op targets. Exact normalized match wins
    (the aux model is shown current_sections, so it can echo exact bullet text);
    loose substring matching is only the fallback when no exact match exists —
    that fallback keeps short human-ish fragments working but can never fire
    when a distinct bullet matches exactly."""
    norm = text.lower().strip().rstrip(".")
    exact = {i for i, b in enumerate(bullets) if _bullet_text(b).lower().strip().rstrip(".") == norm}
    if exact:
        return exact
    return {i for i, b in enumerate(bullets) if norm in b.lower()}


def _bullet_text(bullet: str) -> str:
    """Strip list/checkbox prefixes: '- [ ] question' -> 'question'."""
    return re.sub(r"^\s*-\s*(\[\s*\]\s*)?", "", bullet)


def apply_index_edits(
    idx: IndexFile,
    open_question_ops: Optional[List[dict]] = None,
    timeline_ops: Optional[List[dict]] = None,
    today: Optional[str] = None,
) -> bool:
    """Apply JSON edit ops from the index_edit prompt. Returns True if anything changed."""
    changed = False
    for op in open_question_ops or []:
        action = op.get("action")
        text = (op.get("text") or "").strip()
        if not text:
            continue
        bullets = idx.sections.setdefault("Open Questions", [])
        if action == "add":
            norm = text.lower().rstrip(".")
            if any(norm in b.lower() for b in bullets):
                continue  # idempotent: already present
            bullets.append(f"- [ ] {text}")
            changed = True
        elif action == "resolve":
            hits = _bullet_match_indices(bullets, text)
            new_bullets = [
                re.sub(r"^\s*-\s*\[\s\]\s*", "- [x] ", b) if i in hits else b
                for i, b in enumerate(bullets)
            ]
            if hits:
                changed = True
            idx.sections["Open Questions"] = new_bullets
        elif action == "edit":
            hits = _bullet_match_indices(bullets, text)
            new_bullets = [
                f"- [ ] {op.get('replacement', text)}" if i in hits else b
                for i, b in enumerate(bullets)
            ]
            if hits:
                changed = True
            idx.sections["Open Questions"] = new_bullets
    for op in timeline_ops or []:
        action = op.get("action")
        text = (op.get("text") or "").strip()
        if not text:
            continue
        bullets = idx.sections.setdefault("Timeline of Key Events", [])
        if action == "add":
            norm = text.lower().rstrip(".")
            if any(norm in b.lower() for b in bullets):
                continue  # idempotent: already present
            stamp = op.get("date") or today or date.today().isoformat()
            bullets.append(f"- **{stamp}** — {text}")
            changed = True
        elif action == "edit":
            repl = op.get("replacement")
            hits = _bullet_match_indices(bullets, text) if repl else []
            stamp = op.get("date") or today or date.today().isoformat()
            new_bullets = [
                f"- **{stamp}** — {repl}" if i in hits else b
                for i, b in enumerate(bullets)
            ]
            if hits:
                changed = True
            idx.sections["Timeline of Key Events"] = new_bullets
    return changed
