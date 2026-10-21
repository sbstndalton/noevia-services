"""services#160: another person named in full at the start of an entry or a sentence is another person.

With a full exact hit for the asked two-word person, "Tom Hardy fixed the boat." was kept as if it might be about
the asked person, while "Had coffee with Tom Hardy." was dropped as other_person. Openers that are not people
("Harbour Lights was busy.", "New Year at the cottage.") and the asked person in any form must stay.

Offline only: synthetic rows through the real entity-focus check. No model, embedding or service call.
"""
from __future__ import annotations

import pytest

from agent import entity_focus as ef

ASK = "What did Alder Brannock fix?"
EXACT = "Had a long chat with Alder Brannock about the mooring."


def _row(text, score=0.8, day="2030-01-10"):
    return {"id": text[:8], "day": day, "header": "09:00 — Note", "text": text, "score": score}


def _reasons(question, text, exact=True):
    rows = ([_row(EXACT, 0.82)] if exact else []) + [_row(text, 0.81)]
    res = ef.focus_candidates(question, rows)
    return [rank for rank, _ in res.kept], {rank: why for rank, _, why in res.removed}


# ---------------- A. conflation side ----------------

@pytest.mark.parametrize("text", [
    "Tom Hardy fixed the boat.",
    "Wren Brannocks fixed the boat.",            # the asked surname (plural form), another given name
    "Wren Brannock fixed the boat.",
    "Today Tom Hardy fixed the boat.",
    "Yesterday Tom Hardy fixed the boat.",
    "Later, Tom Hardy fixed the boat.",
    "Rain all morning. Tom Hardy fixed the boat.",   # the start of a later sentence
    "Rain all morning. Today Tom Hardy fixed the boat.",
    "Rain all morning.\n\nTom Hardy fixed the boat.",
    "- Tom Hardy fixed the boat.",
    "“Tom Hardy fixed the boat,” I wrote.",
    "Alder Hollis fixed the boat.",              # the asked given name, another surname
    "Had coffee with Wren Brannocks",            # the mid-sentence form, already dropped
])
def test_another_person_opening_the_entry_or_a_sentence_is_dropped(text):
    assert _reasons(ASK, text) == ([1], {2: "other_person"}), text


# ---------------- B. lost-context side ----------------

@pytest.mark.parametrize("text", [
    "Harbour Lights was busy.",
    "New Year at the cottage.",
    "Saint Ives again.",
    "Bank Holiday walk.",
    "Rain all morning. Harbour Lights was busy.",
    "Today Harbour Lights was busy.",
    "Visited Leeds with the crew.",
    "Met Wren at the quay.",                     # a single word identifies nobody
    "Alder Brannock fixed the boat.",            # the asked person
    "Alder Brannocks boat is fixed.",            # #157: possessive without an apostrophe
    "Alder Brannock’s boat is fixed.",
    "Today Alder Brannock fixed the boat.",
    "Brannock fixed the boat.",                  # the surname alone may be the asked person
])
def test_openers_that_are_not_another_person_are_kept(text):
    kept, removed = _reasons(ASK, text)
    assert kept == [1, 2] and removed == {}, text


def test_a_near_spelling_opening_the_entry_stays_under_the_near_name_rules():
    assert _reasons(ASK, "Alder Brannack fixed the boat.") == ([1], {2: "near_name"})
    # no exact hit: the existing #145 rule (same given name, near surname), unchanged
    assert _reasons(ASK, "Alder Brannack fixed the boat.", exact=False) == ([], {1: "near_name_unmatched"})


# ---------------- C. scope unchanged ----------------

@pytest.mark.parametrize("text", ["Tom Hardy fixed the boat.", "Today Tom Hardy fixed the boat.",
                                  "Wren Brannocks fixed the boat."])
def test_no_drop_without_an_exact_hit(text):
    kept, removed = _reasons(ASK, text, exact=False)
    assert kept == [1] and removed == {}, text


def test_one_word_asked_name_is_unchanged():
    rows = [_row("Had a long chat with Brannock about the mooring.", 0.82), _row("Tom Hardy fixed the boat.", 0.81)]
    res = ef.focus_candidates("What did Brannock fix?", rows)
    assert [r for r, _ in res.kept] == [1, 2] and res.removed == []


def test_relational_question_is_unchanged():
    kept, removed = _reasons("What did Alder Brannock's sister fix?", "Tom Hardy fixed the boat.")
    assert kept == [1, 2] and removed == {}


def test_identified_people_without_asked_names_keeps_the_strict_opener_rule():
    assert ef.identified_people("Tom Hardy fixed the boat.") == []
    asked = [ef.Entity("Alder Brannock", ("alder", "brannock"))]
    assert ef.identified_people("Tom Hardy fixed the boat.", asked=asked) == [("tom", "hardy")]
    assert ef.identified_people("Harbour Lights was busy.", asked=asked) == []


# ---------------- D. hostile inputs ----------------

@pytest.mark.parametrize("text", ["", "   ", "...", "?!—–“”«»", "Tom", "Tom.", "T", "Tom Hardy" + " and" * 4000,
                                  "Tom " * 3000, "“Tom” — “Hardy”", "Ŧom Ħardy fixed it.", "— — Tom — Hardy —",
                                  "\n\n\n", "Today", "Today Today Today Tom", "Tom Hardy"])
def test_hostile_entries_do_not_raise(text):
    kept, _ = _reasons(ASK, text)
    assert 1 in kept


# ---------------- shapes the rule leaves alone on purpose ----------------

def test_a_bold_speaker_label_is_not_a_mention():
    kept, removed = _reasons(ASK, "**Bryony Castellow:** went to Leeds.")
    assert kept == [1, 2] and removed == {}


def test_the_asked_surname_after_a_capitalised_verb_is_not_a_given_name():
    asked = [ef.Entity("Djordje Ilic", ("djordje", "ilic"))]
    assert ef.identified_people("Met Djordje Ilic.", asked=asked) == []


def test_known_limit_a_capitalised_verb_before_another_person_is_kept():
    kept, removed = _reasons(ASK, "Met Tom Hardy at the quay.")  # documented limit (module docstring)
    assert kept == [1, 2] and removed == {}
