"""services#167: a corrected fact is answered with the day of the entry that records it.

Gate run 1effaf5 answered correction questions with a bare current value ("X rents plot 58.") although the
assembled context still labelled every entry with its day and the superseded entry with the correction's day.
The cause was the system prompt rule added with entity focus (#116): "Give the current value only", with no
word about the day. These tests pin both halves: the context carries the days, and the rule asks for them.

Offline only: synthetic rows, a stub retriever and the real assembler and prompt. No model or embedding call.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from agent.config import Config
from agent.context import ContextAssembler

PROMPT = Path(__file__).resolve().parent.parent / "config" / "prompts" / "system.md"
ASK = "Which rowing locker does Odile Marchbank use?"


class FakeStore:
    def get_day_text(self, day, max_chars=None):
        return ""

    def get_standing_sections_text(self, max_chars=None):
        return ""


class StubRetriever:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, top_k=8, min_score=0.3, with_ids=False, diag=None):
        return [dict(r) for r in self.rows[:top_k]]


ROWS = [
    {"id": 1, "day": "2031-04-03", "header": "09:10 — Bumped into Odile Marchbank", "score": 0.81,
     "text": "**Me:** Ran into Odile Marchbank at the boathouse; they told me their rowing locker is locker 12."},
    {"id": 2, "day": "2031-04-09", "header": "21:40 — Correction", "score": 0.80,
     "text": "**Me:** Correction to my note from April 3: Odile Marchbank's rowing locker is actually locker 19, "
             "not locker 12. I had misheard."},
    {"id": 3, "day": "2031-03-28", "header": "13:00 — Chat with Odile Marchbanks", "score": 0.78,
     "text": "**Me:** Odile Marchbanks mentioned the regatta timetable."},
    {"id": 4, "day": "2031-03-02", "header": "18:00 — Bumped into Piers Allardyce", "score": 0.70,
     "text": "**Me:** Piers Allardyce said their rowing locker is locker 3."},
]


def _messages(monkeypatch):
    import agent.workspace_files as wf
    monkeypatch.setattr(wf, "memory_text", lambda store: "")
    cfg = Config({"retrieval": {"top_k": 8, "max_context_tokens": 5200, "min_score": 0.3, "entity_focus": True},
                  "prompts": {"system_path": str(PROMPT)}})
    return ContextAssembler(FakeStore(), StubRetriever(ROWS), cfg).build(date(2031, 5, 1), ASK)


def _rule(text, name):
    m = re.search(r"^- \*\*" + re.escape(name) + r"\.\*\*(.*)$", text, re.M)
    assert m, name
    return m.group(1)


def test_context_carries_the_day_of_the_correction_and_of_the_corrected_entry(monkeypatch):
    block = _messages(monkeypatch)[1]["content"]
    assert "--- [2031-04-09] 21:40 — Correction" in block  # the correction keeps its own day
    original = next(line for line in block.splitlines() if line.startswith("--- [2031-04-03]"))
    assert "correction of 2031-04-09" in original  # and the superseded entry points at that day
    assert "locker 19" in block


def test_latest_record_rule_asks_for_the_day_of_the_current_value(monkeypatch):
    system = _messages(monkeypatch)[0]["content"]
    rule = _rule(system, "The latest record wins")
    assert "current value" in rule
    # The rule must not stop at a bare value: it names the entry day, and for a correction the correction's day.
    assert re.search(r"current value only[^.]*\bday\b", rule), rule
    assert re.search(r"day of the correction", rule), rule
