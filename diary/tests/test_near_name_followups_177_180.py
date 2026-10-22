"""services#177-#180: near-spelled names in follow-ups.

- #177: a name-less follow-up that refers back to the previous exchange ("Is that written in my diary?")
  carries the person of the previous question, so a near-spelled entry is left out as for a named question.
- #178: an entry naming the asked person exactly carries a header label saying so; a near spelling never.
- #179: the follow-up conversation note says to answer from exact entries when one is loaded, and keeps its
  abstain sentence only when none is; never both.
- #180: the system prompt states that an exact-name entry is a record, and how to answer a question that
  adds a detail the entry lacks.

Synthetic only: names come from a small seeded generator (a given name, a surname and a one-letter variant of
that surname), never from a fixture. A stub retriever and the real assembler; no model or embedding call.
"""
from __future__ import annotations

import random
import re
from datetime import date
from pathlib import Path

import pytest

from agent import entity_focus as ef
from agent import follow_up
from agent.config import Config
from agent.context import ContextAssembler

PROMPT = Path(__file__).resolve().parent.parent / "config" / "prompts" / "system.md"
BEGIN, END = "=== RETRIEVED PAST ENTRIES", "=== END RETRIEVED PAST ENTRIES ==="
TODAY = date(2030, 3, 10)
META = "Is that written in my diary, or are you inferring it?"
EXACT_LABEL = "names the asked person"
ANSWER_FROM = "answer from them"
ABSTAIN = "If the diary records nothing"

_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "br", "dr", "gr", "kl", "st", "tr"]
_VOWELS = ["a", "e", "i", "o", "u"]
_CODAS = ["", "n", "r", "l", "s", "m", "th", "ck"]


def _word(rng, syllables):
    return "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS) for _ in range(syllables)).title()


def _variant(rng, surname):
    """A one-letter substitution inside the surname that entity focus reads as a near spelling."""
    for _ in range(200):
        i = rng.randrange(1, len(surname) - 1)
        pool = _VOWELS if surname[i] in _VOWELS else [c for c in "bdfgklmnprstv"]
        new = surname[:i] + rng.choice([c for c in pool if c != surname[i]]) + surname[i + 1:]
        if ef.near_word(new, surname) and new.lower() != surname.lower() + "s":
            return new
    raise AssertionError("no variant")


def _pairs(n=8, seed=177):
    """(given, surname, near surname) triples; surnames are at least six letters."""
    rng, out = random.Random(seed), []
    while len(out) < n:
        given, surname = _word(rng, 2), _word(rng, 2)
        if len(surname) < 6 or len(given) < 4 or given.lower() in ef._NOT_NAMES:
            continue
        out.append((given, surname, _variant(rng, surname)))
    return out


PAIRS = _pairs()


def _q(name):
    return f"What kite colour does {name} fly?"


def _row(i, day, name, value, score):
    return {"id": i, "day": day, "header": "10:00 — Coffee", "score": score,
            "text": f"Had coffee with {name} by the quay; they told me their kite colour is {value}."}


class _Store:
    index_enabled = False

    def get_day_text(self, day, max_chars=None):
        return ""

    def get_standing_sections_text(self, max_chars=None):
        return ""


class _Retriever:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, top_k=8, min_score=0.3, with_ids=False, diag=None):
        return [dict(r) for r in sorted(self.rows, key=lambda r: -r["score"])[:top_k]]


@pytest.fixture(autouse=True)
def _no_memory(monkeypatch):
    import agent.workspace_files as wf
    monkeypatch.setattr(wf, "memory_text", lambda store: "")


def _build(message, turns, rows):
    cfg = Config({"context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
                  "retrieval": {"top_k": 8, "min_score": 0.3, "max_context_tokens": 5200, "entity_focus": True},
                  "prompts": {"system_path": ""}})
    asm = ContextAssembler(_Store(), _Retriever(rows), cfg)
    asm._system_prompt = lambda: "SYSTEM"
    messages = asm.build(TODAY, message, session_turns=turns)
    block = messages[1]["content"]
    return messages, block, block[block.index(BEGIN):block.index(END)]


def _turns(*pairs):
    out = []
    for question, answer in pairs:
        out += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return out


def _header(retrieved, value):
    """The header line of the loaded entry holding ``value``."""
    lines = retrieved.splitlines()
    i = next(i for i, line in enumerate(lines) if value in line)
    return lines[i - 1]


# ---------------- #177: name-less follow-ups inherit the asked person ----------------

@pytest.mark.parametrize("given,surname,near", PAIRS)
@pytest.mark.parametrize("message", [META, "Where is that recorded?", "Are you sure about it?"])
def test_a_name_less_follow_up_carries_the_previous_person(given, surname, near, message):
    asked = f"{given} {surname}"
    res = ef.analyse(message, [_q(asked)])
    assert res.carried and [e.name for e in res.entities] == [asked]


@pytest.mark.parametrize("given,surname,near", PAIRS)
def test_the_near_spelled_entry_is_left_out_of_a_name_less_follow_up(given, surname, near):
    asked, other = f"{given} {surname}", f"{given} {near}"
    rows = [_row(1, "2030-01-24", other, "teal", 0.70), _row(2, "2030-01-10", asked, "saffron", 0.69)]
    turns = _turns((_q(asked), f"Your entry from January 10, 2030 says {asked}'s kite colour is saffron."))
    messages, block, retrieved = _build(META, turns, rows)
    assert messages[-1]["content"] == META
    assert "teal" not in retrieved and other not in retrieved
    assert "saffron" in retrieved and f'"{asked}"' in _header(retrieved, "saffron")
    assert "carried from the earlier question" in retrieved


@pytest.mark.parametrize("given,surname,near", PAIRS[:4])
def test_a_message_naming_someone_new_never_inherits(given, surname, near):
    asked, new = f"{given} {surname}", f"{PAIRS[-1][0]} {PAIRS[-1][1]}"
    res = ef.analyse(f"Is that written about {new} too?", [_q(asked)])
    assert not res.carried and [e.name for e in res.entities] == [new]


@pytest.mark.parametrize("message", ["How has the week been?", "How has this week been?",
                                     "What did I do that morning?", "Summarise my month."])
def test_a_message_referring_back_to_nothing_carries_nothing(message):
    given, surname, _ = PAIRS[0]
    assert not ef.analyse(message, [_q(f"{given} {surname}")]).active


@pytest.mark.parametrize("given,surname,near", PAIRS[:4])
def test_a_chained_follow_up_carries_the_latest_named_person(given, surname, near):
    first, second = f"{given} {surname}", f"{given} {near}"
    rows = [_row(1, "2030-01-24", first, "teal", 0.70), _row(2, "2030-01-10", second, "saffron", 0.69)]
    turns = _turns((_q(first), "Synthetic answer about teal."), (f"And {second}?", "Synthetic answer."))
    _, _, retrieved = _build(META, turns, rows)
    assert "saffron" in retrieved and "teal" not in retrieved  # the latest question named the near spelling
    # #176 is unchanged: a further elliptical follow-up after an elliptical one is not resolved.
    third = f"{given} {_variant(random.Random(1), near)}"
    messages, block, _ = _build(f"And {third}?", turns, rows)
    assert messages[-1]["content"] == f"And {third}?" and "[Reference note on this conversation" not in block


# ---------------- #178: exact entries are labelled, near ones never ----------------

@pytest.mark.parametrize("given,surname,near", PAIRS)
def test_the_exact_entry_is_labelled_and_the_near_one_never(given, surname, near):
    asked, other = f"{given} {surname}", f"{given} {near}"
    exact, nearby = _row(2, "2030-01-10", asked, "saffron", 0.69), _row(1, "2030-01-24", other, "teal", 0.70)
    res = ef.focus_candidates(_q(asked), [nearby, exact])
    assert ef.entry_label(res, 2) == f'; names the asked person "{asked}" exactly'
    assert [(rank, reason) for rank, _, reason in res.removed] == [(1, "near_name")]
    alone = ef.focus_candidates(_q(asked), [nearby])  # no exact entry: #145 leaves it out, never labelled exact
    assert EXACT_LABEL not in ef.entry_label(alone, 1) and not alone.exact
    assert [reason for _, _, reason in alone.removed] == ["near_name_unmatched"]
    other_given = ef.focus_candidates(_q(asked), [_row(1, "2030-01-24", f"{given}x {near}", "teal", 0.70)])
    if other_given.kept:  # kept while nobody is named exactly: hedged, never labelled exact
        assert EXACT_LABEL not in ef.entry_label(other_given, 1)
    _, _, retrieved = _build(_q(asked), None, [nearby, exact])
    assert EXACT_LABEL in _header(retrieved, "saffron") and "teal" not in retrieved


@pytest.mark.parametrize("given,surname,near", PAIRS[:4])
def test_accented_and_differently_cased_exact_names_are_labelled(given, surname, near):
    accented = given[0] + given[1:].replace("a", "á", 1).replace("e", "ë", 1)
    asked = f"{accented} {surname}"
    for written in (f"{given} {surname}", f"{given.upper()} {surname.upper()}", asked):
        rows = [_row(1, "2030-01-24", f"{given} {near}", "teal", 0.70), _row(2, "2030-01-10", written, "saffron", 0.69)]
        res = ef.focus_candidates(_q(asked), rows)
        assert EXACT_LABEL in ef.entry_label(res, 2), written
        assert 1 not in res.exact and (1, rows[0], "near_name") in res.removed


def test_an_entry_naming_only_a_first_name_gets_no_exact_label():
    given, surname, _ = PAIRS[0]
    rows = [{"id": 1, "day": "2030-01-10", "header": "10:00", "score": 0.6, "text": f"{given} rang about the kite."}]
    res = ef.focus_candidates(_q(f"{given} {surname}"), rows)
    assert ef.entry_label(res, 1) == ""


# ---------------- #179: the conversation note's form follows what was loaded ----------------

@pytest.mark.parametrize("given,surname,near", PAIRS)
def test_the_note_depends_only_on_whether_an_exact_entry_was_loaded(given, surname, near):
    first, second = f"{given} {surname}", f"{given} {near}"
    turns = _turns((_q(first), f"Your entry from January 10, 2030 says {first}'s kite colour is saffron."))
    loaded = [_row(1, "2030-01-10", first, "saffron", 0.70), _row(2, "2030-02-02", second, "teal", 0.69)]
    _, block, retrieved = _build(f"And what about {second}?", turns, loaded)
    assert "teal" in retrieved
    note = block[block.index("[Reference note on this conversation"):]
    assert ANSWER_FROM in note and f'name "{second}" exactly' in note and ABSTAIN not in note
    _, block, _ = _build(f"And what about {second}?", turns, loaded[:1])  # only the earlier person's entry
    note = block[block.index("[Reference note on this conversation"):]
    assert ABSTAIN in note and ANSWER_FROM not in note


@pytest.mark.parametrize("near", [True, False])
def test_the_note_never_holds_both_forms(near):
    given, surname, variant = PAIRS[1]
    res = follow_up.Resolution(standalone="q", earlier_name=f"{given} {surname}", asked_name=f"{given} {variant}",
                               near=near)
    with_exact, without = follow_up.conversation_note(res, True), follow_up.conversation_note(res, False)
    assert ANSWER_FROM in with_exact and ABSTAIN not in with_exact
    assert ANSWER_FROM not in without and (ABSTAIN in without) == near


# ---------------- #180: the system prompt's answer rules ----------------

def _rule(text, name):
    m = re.search(r"^- \*\*" + re.escape(name) + r"\.\*\*(.*)$", text, re.M)
    assert m, name
    return m.group(1)


def test_system_prompt_states_the_exact_record_and_missing_detail_rules():
    text = PROMPT.read_text(encoding="utf-8")
    exact = _rule(text, "An exact name is a record")
    assert "Never say the diary does not record" in exact and "exact person" in exact
    detail = _rule(text, "Missing detail, not missing fact")
    assert "weekday" in detail and "recorded value" in detail and "day of the entry" in detail
    assert "not recorded" in detail and "rather than abstaining" in detail
    # The earlier rules stay, in order, ahead of the new ones where they belong.
    for rule in ("Keep people apart", "Similar names: check, do not assume", "The latest record wins",
                 "Withdrawn means not recorded", "Abstain plainly", "Time-anchored questions", "Shared first names"):
        _rule(text, rule)
    assert "day of the correction" in _rule(text, "The latest record wins")  # services#170
    assert text.index("Abstain plainly") < text.index("An exact name is a record") < text.index("Time-anchored")
    assert "[LOG: skip]" in text and "[LOG: ok]" in text
