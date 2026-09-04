"""CorpusStore — owns the diary corpus: WebDAV reads/writes + write-ahead journal.

Every mutation is: enqueue in journal → apply via ETag-guarded WebDAV PUT → mark applied.
Replays after crashes are safe: month appends dedupe on the xid marker; index edits
dedupe on bullet-text equality.
"""
from __future__ import annotations

import calendar
import logging
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import corpus as fmt
from .config import Config
from .journal import Journal, JournalEntry
from .webdav import WebDAVClient

log = logging.getLogger(__name__)


class CorpusError(RuntimeError):
    pass


class CorpusStore:
    def __init__(self, cfg: Config, dav: WebDAVClient, journal: Journal):
        self.cfg = cfg
        self.dav = dav
        self.journal = journal
        self.remote_root = (cfg.get("corpus.webdav.remote_root") or "").strip("/")
        self.monthly_prefix = cfg.get("corpus.monthly_prefix") or ""
        self.index_file = cfg.get("corpus.index_file") or "INDEX.md"
        # Month-file naming template (default preserves the original 2026-09 style).
        # Example for human-named corpora: "Diary - {month_name} {year}.md"
        self.month_file_template = cfg.get("corpus.month_file_template") or "{year}-{month:02d}.md"
        # INDEX.md standing sections can be disabled entirely (corpora that manage
        # their own index / don't use one). Default: enabled (original behavior).
        self.index_enabled = bool(cfg.get("corpus.index_enabled", True))

    # ---------------- paths ----------------

    def month_filename(self, day: date) -> str:
        return self.month_file_template.format(
            year=day.year,
            month=day.month,
            month02=f"{day.month:02d}",
            month_name=day.strftime("%B"),
        )

    def _join(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in [self.remote_root, *parts] if p)

    def month_path(self, day: date) -> str:
        return self._join(self.monthly_prefix, self.month_filename(day))

    def index_path(self) -> str:
        return self._join(self.monthly_prefix, self.index_file)

    # ---------------- reads ----------------

    def read_month(self, day: date) -> Tuple[Optional[str], Optional[str]]:
        return self.dav.get_text(self.month_path(day))

    def read_index(self) -> Tuple[Optional[str], Optional[str]]:
        return self.dav.get_text(self.index_path())

    def month_label(self, day: date) -> str:
        return day.strftime("%B %Y")

    # ---------------- month browsing (read-only) ----------------

    def read_month_text(self, year: int, month: int) -> str:
        """A whole month file's text, xid markers stripped, for display.

        Empty string when the month has no file yet (or the read fails —
        display-only read, same degradation policy as get_day_text).
        """
        try:
            month_text, _ = self.read_month(date(year, month, 1))
        except Exception as exc:  # noqa: BLE001
            log.warning("month-text read failed (degrading to empty): %s", exc)
            return ""
        if not month_text:
            return ""
        return fmt.strip_markers(month_text)

    def list_months(self) -> List[dict]:
        """Months that actually have a corpus file, oldest first.

        Walks the monthly dir via PROPFIND and matches filenames against the
        configured month_file_template ({year}, {month02}, {month_name} are
        recognized). Non-matching files and subdirectories are ignored.
        """
        template = self.month_file_template
        # Build a regex from the template: literal text around named fields.
        pattern = re.escape(template)
        pattern = pattern.replace(re.escape("{year}"), r"(?P<year>\d{4})")
        pattern = pattern.replace(re.escape("{month02}"), r"(?P<month02>\d{2})")
        pattern = pattern.replace(re.escape("{month}"), r"(?P<month>\d{1,2})")
        pattern = pattern.replace(re.escape("{month_name}"), r"(?P<month_name>[A-Za-z]+)")
        pattern = f"^{pattern}$"
        rx = re.compile(pattern)

        months: List[dict] = []
        seen = set()
        try:
            entries = self.dav.list_dir(self._join(self.monthly_prefix))
        except Exception as exc:  # noqa: BLE001
            log.warning("month listing failed (degrading to empty): %s", exc)
            return []
        for entry in entries:
            if entry.get("is_dir"):
                continue
            name = entry.get("name") or ""
            if name == self.index_file:
                continue
            m = rx.match(name)
            if not m:
                continue
            gd = m.groupdict()
            try:
                year = int(gd["year"])
                if gd.get("month_name"):
                    month = next(
                        (i for i, mn in enumerate(calendar.month_name) if mn.lower() == gd["month_name"].lower()),
                        None,
                    )
                    if month is None:
                        continue
                elif gd.get("month02"):
                    month = int(gd["month02"])
                elif gd.get("month"):
                    month = int(gd["month"])
                else:
                    month = 1  # lone {year} template
            except (TypeError, ValueError):
                continue
            if not (1 <= month <= 12) or year < 2000 or year > 2100:
                continue
            key = (year, month)
            if key in seen:
                continue
            seen.add(key)
            label = date(year, month, 1).strftime("%B %Y")
            months.append({"id": f"{year:04d}-{month:02d}", "label": label, "file": name})
        months.sort(key=lambda x: x["id"])
        return months

    # ---------------- guarded remote writes ----------------

    def _guarded_write(
        self,
        path: str,
        mutate: Callable[[Optional[str]], Tuple[Optional[str], bool]],
        max_attempts: int = 5,
    ) -> Tuple[bool, Optional[str]]:
        """ETag-guarded create/modify loop. mutate(current_text) -> (new_text, changed).

        On 412 (concurrent change), re-GETs and re-runs mutate against fresh content —
        the mutate function must be idempotent w.r.t. already-applied content.
        Returns (applied, final_etag).
        """
        for attempt in range(max_attempts):
            text, etag = self.dav.get_text(path)
            new_text, changed = mutate(text)
            if not changed:
                return True, etag  # nothing to do (already applied)
            ok, new_etag, status = self.dav.put(
                path,
                (new_text or "").encode("utf-8"),
                if_match=etag,
            )
            if ok:
                return True, new_etag
            log.warning("write conflict on %s (status %s), attempt %d", path, status, attempt + 1)
        raise CorpusError(f"could not write {path} after {max_attempts} attempts (persistent conflict)")

    # ---------------- exchange logging ----------------

    def log_exchange(
        self,
        day: date,
        sub_header: str,
        me_text: str,
        claude_text: str,
        now: Optional[datetime] = None,
    ) -> str:
        """Durably record the intent to log an exchange, then apply it. Returns the xid."""
        xid = str(uuid.uuid4())
        body = fmt.render_exchange(me_text, claude_text, xid)
        rendered_header = fmt.render_subsection_header(now=now, topic=sub_header)
        # store the header TEXT (no '### ' prefix) — the appender adds the prefix
        header_text = rendered_header[4:] if rendered_header.startswith("### ") else rendered_header
        self.journal.enqueue(
            "exchange",
            {
                "xid": xid,
                "day": day.isoformat(),
                "sub_header": header_text,
                "body": body,
                "month": self.month_filename(day),
                "month_label": self.month_label(day),
            },
        )
        # Register the month link too (applier dedupes; harmless if the link exists).
        # Skipped entirely when the corpus runs without an INDEX.md.
        if self.index_enabled:
            self.journal.enqueue(
                "index_month",
                {"month": self.month_filename(day), "label": self.month_label(day)},
            )
        self.apply_pending()
        return xid

    def update_standing_sections(self, open_question_ops: list, timeline_ops: list, today: str) -> Optional[str]:
        """Enqueue + apply INDEX.md standing-section edits. Returns journal id."""
        if not self.index_enabled:
            return None
        if not open_question_ops and not timeline_ops:
            return None
        jid = self.journal.enqueue(
            "index_update",
            {"open_questions": open_question_ops, "timeline": timeline_ops, "today": today},
        )
        self.apply_pending()
        return jid

    # ---------------- appliers ----------------

    def apply_pending(self, limit: int = 100) -> int:
        """Apply all unapplied journal entries in order. Returns count applied now."""
        applied = 0
        for entry in self.journal.unapplied(limit=limit):
            try:
                self._apply_entry(entry)
                self.journal.mark_applied(entry.id)
                applied += 1
            except Exception as exc:  # noqa: BLE001 — keep trying remaining entries
                log.error("journal entry %s failed: %s", entry.id, exc)
                self.journal.mark_failed(entry.id, str(exc))
        return applied

    def _apply_entry(self, entry: JournalEntry) -> None:
        if entry.kind == "exchange":
            self._apply_exchange(entry)
        elif entry.kind == "index_month":
            self._apply_month_registration(entry)
        elif entry.kind == "index_update":
            self._apply_index_update(entry)
        else:
            raise CorpusError(f"unknown journal kind: {entry.kind}")

    def _apply_exchange(self, entry: JournalEntry) -> None:
        p = entry.payload
        day = date.fromisoformat(p["day"])
        xid = p["xid"]
        sub_header = p["sub_header"]
        body = p["body"]
        path = self.month_path(day)

        def mutate(current: Optional[str]) -> Tuple[Optional[str], bool]:
            if current is not None and fmt.has_marker(current, xid):
                return current, False  # already applied — replay safe
            return fmt.append_to_month_text(current, day, sub_header, body), True

        self._guarded_write(path, mutate)
        log.info("exchange %s logged to %s", xid, path)

    def _apply_month_registration(self, entry: JournalEntry) -> None:
        p = entry.payload

        def mutate(current: Optional[str]) -> Tuple[Optional[str], bool]:
            idx = fmt.parse_index(current)
            if fmt.register_month(idx, p["label"], p["month"]):
                return idx.render(), True
            return current, False

        self._guarded_write(self.index_path(), mutate)

    def _apply_index_update(self, entry: JournalEntry) -> None:
        p = entry.payload

        def mutate(current: Optional[str]) -> Tuple[Optional[str], bool]:
            idx = fmt.parse_index(current)
            changed = fmt.apply_index_edits(
                idx,
                open_question_ops=p.get("open_questions") or [],
                timeline_ops=p.get("timeline") or [],
                today=p.get("today"),
            )
            return (idx.render(), True) if changed else (current, False)

        self._guarded_write(self.index_path(), mutate)

    # ---------------- context helpers ----------------

    def get_day_text(self, day: date, max_chars: Optional[int] = None) -> str:
        """The current day's log section, markers stripped, for the model context.

        If over max_chars, the earliest exchanges are dropped (recency matters most
        within a day); a truncation notice is prepended. Read failures (e.g. WebDAV
        misconfigured or Nextcloud briefly down) degrade to empty context — the chat
        keeps working; WRITES never degrade and will surface errors loudly.
        """
        try:
            month_text, _ = self.read_month(day)
        except Exception as exc:  # noqa: BLE001
            log.warning("day-text read failed (degrading to empty): %s", exc)
            return ""
        if not month_text:
            return ""
        header = fmt.canonical_day_header(day)
        for d in fmt.parse_diary(month_text):
            if d.header == header:
                parts = [d.header, ""]
                for sub in d.subsections:
                    parts.append(f"### {sub.header}")
                    parts.append("")
                    for ex in sub.exchanges:
                        if ex.me:
                            parts.append(f"**Me:** {ex.me}")
                            parts.append("")
                        if ex.claude:
                            parts.append(f"**Claude:** {ex.claude}")
                            parts.append("")
                text = "\n".join(parts).strip()
                if max_chars and len(text) > max_chars:
                    text = "(earlier exchanges today truncated)\n…" + text[-max_chars:]
                return text
        return ""

    def get_standing_sections_text(self, max_chars: Optional[int] = None) -> str:
        if not self.index_enabled:
            return ""
        try:
            index_text, _ = self.read_index()
        except Exception as exc:  # noqa: BLE001
            log.warning("index read failed (degrading to empty): %s", exc)
            index_text = None
        idx = fmt.parse_index(index_text)
        out = fmt.IndexFile(
            preamble=idx.preamble, month_links=idx.month_links, sections=idx.sections
        ).render()
        if max_chars and len(out) > max_chars:
            out = out[:max_chars] + "\n…(standing sections truncated)"
        return out

    def month_registered(self, day: date) -> bool:
        index_text, _ = self.read_index()
        idx = fmt.parse_index(index_text)
        return any(self.month_filename(day) in link for link in idx.month_links)

    def new_xid(self) -> str:
        return str(uuid.uuid4())
