"""services#116: near-name precision, the exact-entity context and the deletion-abstain context.

Synthetic only: the gate's own fixture generator (diary/gate/gen_fixture.py) plus small invented cases, a
character-trigram stub embedder (similar spellings embed close together, which is how the failure looks with
a real embedder) and no model or inference call. Precision and recall use the gate's own scorer and
thresholds (diary/gate/score.py, thresholds.json) on the opt-in retrieval trace.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import corpus as fmt
from agent import entity_focus as ef
from agent.config import Config
from agent.context import ContextAssembler
from agent.retrieval import Retriever
from agent.retrieval_trace import RetrievalTrace
from tests.test_retrieval_context import FakeStore

GATE = Path(__file__).resolve().parent.parent / "gate"
sys.path.insert(0, str(GATE))

import gen_fixture  # noqa: E402
import score  # noqa: E402

THRESHOLDS = json.loads((GATE / "thresholds.json").read_text())
K = THRESHOLDS["candidate_retrieval"]["k"]
SEED, ANCHOR, DAYS = 20261020, date(2030, 1, 6), 60
BEGIN, END = "=== RETRIEVED PAST ENTRIES", "=== END RETRIEVED PAST ENTRIES ==="


class TrigramLLM:
    """Deterministic embedder: hashed character trigrams of each word, L2-normalised. No model."""

    DIM = 256

    def embed(self, texts, model=None):
        out = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                word = f"^{word}$"
                for i in range(len(word) - 2):
                    h = int.from_bytes(hashlib.blake2b(word[i:i + 3].encode(), digest_size=4).digest(), "big")
                    vec[h % self.DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def chat(self, *a, **k):  # pragma: no cover
        raise AssertionError("no inference in these tests")


def _cfg(focus=True, min_score=0.30):
    # The #114 score gap is off: "focus off" here means plain vector search, the baseline the focus is measured
    # against. tests/test_score_gap.py covers the gap.
    return Config({
        "context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
        "retrieval": {"top_k": 8, "min_score": min_score, "max_context_tokens": 5200, "entity_focus": focus,
                      "score_gap": "off"},
        "prompts": {"system_path": ""},
    })


def _assembler(retriever, focus=True, min_score=0.30):
    asm = ContextAssembler(FakeStore(), retriever, _cfg(focus, min_score))
    asm._system_prompt = lambda: "SYSTEM"
    return asm


def _retrieved_block(messages) -> str:
    content = messages[1]["content"]
    return content[content.index(BEGIN):content.index(END)]


def _index_files(retriever, files):
    for path, text in sorted(files.items()):
        retriever.reindex_file(path, text)
    if not retriever.vec_available:
        pytest.skip("sqlite-vec not available in this environment")


def _apply(files, entries, mutations):
    """The gate's scripted mutations, applied to the synthetic files as the API would write them."""
    files = dict(files)
    path_of = {e["xid"]: e["path"] for e in entries}
    for m in mutations:
        if m["op"] == "edit":
            new = fmt.replace_exchange_text(files[path_of[m["xid"]]], m["xid"], m["me"], m["assistant"])
            assert new is not None
            files[path_of[m["xid"]]] = new
        else:
            files[m["path"]] = files[m["path"]].rstrip("\n") + "\n\n" + m["block"] + "\n"
    return files


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    files, entries, queries, mutations, _ = gen_fixture.build(SEED, ANCHOR, DAYS)
    files = _apply(files, entries, mutations)
    retriever = Retriever(tmp_path_factory.mktemp("focus") / "idx.db", TrigramLLM())
    _index_files(retriever, files)
    index = {(e["day"], e["header"][:5]): e["id"] for e in entries}
    assert len(index) == len(entries)  # day + clock time names each synthetic entry
    return SimpleNamespace(retriever=retriever, index=index, queries=queries,
                           by_id={q["id"]: q for q in queries}, entries={e["id"]: e for e in entries})


def ask(world, text, day, turns=None, focus=True):
    trace = RetrievalTrace()
    messages = _assembler(world.retriever, focus).build(day, text, session_turns=turns, trace=trace)
    ids = lambda rows: [world.index.get((r["day"], r["time"]), "unmapped") for r in rows]  # noqa: E731
    return SimpleNamespace(messages=messages, block=_retrieved_block(messages), trace=trace.data,
                           candidates=ids(trace.data["candidates"]), loaded=ids(trace.data["loaded"]))


def _day(q):
    return date.fromisoformat(q["clock"][:10])


def metrics(world, focus, categories=None):
    rec, prec, ctx = [], [], []
    for q in world.queries:
        if not q["relevant_entry_ids"] or (categories and q["category"] not in categories):
            continue
        got = ask(world, q["text"], _day(q), focus=focus)
        sets, top = q["relevant_entry_ids"], got.candidates[:K]
        rec.append(score.best_recall(sets, top))
        prec.append(score.precision(sets, top))
        ctx.append(score.best_recall(sets, got.loaded))
    return {"queries": len(rec), "recall": score.mean(rec), "precision": score.mean(prec),
            "context_recall": score.mean(ctx)}


def _text(retriever, chunk_id):
    return retriever._conn.execute("SELECT body FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]


def _loaded_kinds(world, got, entity):
    """How each loaded entry relates to ``entity`` (entity_focus.classify), from the indexed text."""
    return [ef.classify(_text(world.retriever, e["id"]), entity)[0] for e in got.trace["loaded"]]


def _norm(text):
    return re.sub(r"\s+", " ", text.lower())


def _names(entities):
    return [e.name for e in entities]


# ---------------- the lexical check ----------------


@pytest.mark.parametrize("message, entities", [
    ("What is Alder Brannock's favourite tea?", ["Alder Brannock"]),
    ("What is Alder Brannock’s favourite tea?", ["Alder Brannock"]),
    ("And what about Alder Brennock?", ["Alder Brennock"]),
    ("Which three places did I survey during the Hollin Moor survey?", ["Hollin Moor"]),
    ("What did Tobiah give me?", ["Tobiah"]),
    ("Alder Brannock came by. How was I feeling?", ["Alder Brannock"]),
    ("Did Bryony and Caspian Dunmore meet on Monday in March?", ["Bryony", "Caspian Dunmore"]),
    ("Which ferry crossing does Hesper Gilchrist take on Thursdays?", ["Hesper Gilchrist"]),
    ("What did Dr. Patel say?", ["Dr Patel"]),
    ("Which dentist did I see in February?", []),
    ("Tell me about the entries", []),
    ("Remind me what I wrote yesterday.", []),
    ("SECRETQUERYPHRASE about the entries", []),
    ("", []),
])
def test_asked_entities(message, entities):
    assert _names(ef.asked_entities(message)) == entities


def test_asked_entities_are_capped_and_unique():
    found = ef.asked_entities("Did Anna, anna, Bea, Cleo, Dora and Esme come?")
    assert _names(found) == ["Anna", "Bea", "Cleo", "Dora"][:ef.MAX_ENTITIES]
    titled = ef.asked_entities("Did Dr Patel or Dr. Patel call?")
    assert len(titled) == 1 and titled[0].keys == ("patel",)


@pytest.mark.parametrize("message, anchor", [
    ("As of January 20, 2030, what had I written down as Alder Brannock's tea?", (date(2030, 1, 20), True)),
    ("as at 3rd February 2030 what was it?", (date(2030, 2, 3), True)),
    ("Before 2030-02-10, what did I note?", (date(2030, 2, 10), False)),
    ("Up to Sept 9, 2030, who visited?", (date(2030, 9, 9), True)),
    ("As of January 20, 2030, what did Dr. Patel say?", (date(2030, 1, 20), True)),
    ("As of today, how am I?", None),
    ("As of February 30, 2030, what then?", None),  # not a date: no anchor rather than an error
    ("What happened on January 20, 2030?", None),
    # Dates inside statements or in the middle of a question are not anchors (review finding 3).
    ("Sam is away until 3 March 2030. Is Sam back yet?", None),
    ("What did I note before 2030-02-10?", None),
    ("As of 3 March 2030 I live in Leeds. Where did Sam move?", None),
    ("Is Sam back? Sam is away until 3 March 2030.", None),
])
def test_time_anchor(message, anchor):
    assert ef.time_anchor(message) == anchor


def _ent(name):
    found = ef.asked_entities(f"Is it about {name}?")
    assert len(found) == 1, name
    return found[0]


@pytest.mark.parametrize("text, name, kind", [
    ("Ran into Alder Brannock at the market.", "Alder Brannock", "match"),
    ("Alder called, new number is 0789.", "Alder Brannock", "partial"),  # first name on its own
    ("Brannock rang about the boat.", "Alder Brannock", "partial"),  # surname on its own
    ("Dr. Brannock rang.", "Alder Brannock", "partial"),
    ("Saw Dr. Patel about my knee.", "Dr Patel", "match"),
    ("Saw Dr Patel about my knee.", "Dr. Patel", "match"),
    ("José García visited.", "Jose Garcia", "match"),
    ("Mary-Jane Okafor phoned.", "Mary Jane Okafor", "match"),
    ("Had coffee with Alder Brennock.", "Alder Brannock", "near"),
    ("Dr Patil said to rest it.", "Dr Patel", "near"),
    ("Al Brannock sailed today.", "Alder Brannock", "variant"),  # a possible short form, never dropped
    ("Alder Pennick came round.", "Alder Brannock", "shared"),
    ("Bryony Castellow came round.", "Alder Brannock", None),
    ("the brennock was cold", "Brannock", None),  # a lower-case word is not a name
])
def test_classify(text, name, kind):
    assert ef.classify(text, _ent(name))[0] == kind


def test_every_fixture_near_name_pair_is_near():
    for real, near in gen_fixture.SURNAMES:
        assert ef.near_word(real, near) and ef.near_word(near, real), (real, near)
        assert ef.classify(f"Had coffee with Alder {near} today.", _ent(f"Alder {real}"))[0] == "near"
    assert ef.near_word("Ivo", "Iva") and not ef.near_word("Ivo", "Ida")  # short names: one edit at most
    assert not ef.near_word("Al", "Ab")  # under three letters: never


def _row(day, header, text, score=0.5):
    return {"day": day, "header": header, "text": text, "score": score}


TEA = [
    _row("2030-01-10", "09:00 — Chat with Alder Brennock", "Alder Brennock likes nettle tea.", 0.9),
    _row("2030-01-08", "12:00 — Bumped into Alder Brannock", "Alder Brannock said his tea is smoked lapsang.", 0.8),
    _row("2030-01-09", "19:00 — Studio day", "Inked the coastal chart.", 0.78),  # within MARGIN of 0.8
    _row("2030-01-30", "22:00 — Correction",
         "Correction to my note: Alder Brannock's tea is white peony, not smoked lapsang.", 0.6),
    _row("2030-01-11", "08:00 — Phone", "Alder called, new number is 0789.", 0.55),
]


def test_focus_drops_only_confirmed_near_names():
    res = ef.focus_candidates("What is Alder Brannock's favourite tea?", TEA)
    assert res.applied and [rank for rank, _ in res.kept] == [2, 3, 4, 5]  # unnamed and first-name entries stay
    assert [(rank, reason) for rank, _, reason in res.removed] == [(1, "near_name")]
    assert res.corrected_by == {2: "2030-01-30"}  # the entry holding the corrected value
    assert "dated before the correction of 2030-01-30" in ef.entry_label(res, 2)
    assert ef.entry_label(res, 5) == "" and ef.entry_label(res, 4) == ""
    note = ef.context_notes(res)[0]
    assert '"Alder Brannock"' in note and "near spelling of that name were left out" in note
    assert "do not answer" not in note.lower()

    anchored = ef.focus_candidates("As of January 20, 2030, what was Alder Brannock's tea?", TEA)
    assert 4 not in [rank for rank, _ in anchored.kept] and (4, TEA[3], "after_time_anchor") in anchored.removed
    assert anchored.corrected_by == {}
    assert "Time anchor" in ef.context_notes(anchored)[0]


def test_focus_without_an_exact_entry_drops_nothing_and_hedges():
    rows = [TEA[0], TEA[2]]
    res = ef.focus_candidates("What is Alder Brannock's favourite tea?", rows)
    assert not res.applied and [rank for rank, _ in res.kept] == [1, 2] and res.removed == []
    assert ef.entry_label(res, 1) == ('; names "Alder Brennock", similar to "Alder Brannock": '
                                      "may be a different person, check the name")
    note = ef.context_notes(res)[0]
    assert "No retrieved entry names" in note and "check the name" in note and "do not answer" not in note.lower()


def test_typo_in_the_question_never_hides_the_entry():
    """Review finding 2: "Alder Branock" matches nothing exactly, so both near spellings stay, hedged."""
    res = ef.focus_candidates("What is Alder Branock's favourite tea?", TEA)
    assert res.removed == [] and len(res.kept) == len(TEA)
    assert all("may be a different person, check the name" in ef.entry_label(res, r) for r in (1, 2, 4))


def test_nickname_in_the_question_or_the_entry_is_never_dropped():
    rows = [_row("2030-01-08", "12:00 — Sailing", "Al Brannock took the boat out."),
            _row("2030-01-09", "12:00 — Tea", "Alder Brannock drinks smoked lapsang.")]
    res = ef.focus_candidates("What does Alder Brannock drink?", rows)
    assert res.removed == [] and "may be a different person" in ef.entry_label(res, 1)
    asked_short = ef.focus_candidates("What does Al Brannock drink?", rows)
    assert asked_short.removed == [] and len(asked_short.kept) == 2


def test_incidental_or_unrelated_corrections_supersede_nothing():
    """Review finding 4: only an explicit correction of the same entity that replaces a value in the entry."""
    base = _row("2030-01-01", "09:00 — Tea", "Alder Brannock's favourite tea is smoked lapsang.")
    cases = [
        _row("2030-01-05", "10:00 — Marking", "Corrected the essays with Alder Brannock all afternoon."),
        _row("2030-01-05", "10:00 — Correction", "Correction: Alder Brannock's bicycle is green, not blue."),
        _row("2030-01-05", "10:00 — Correction", "Correction: Bryony Castellow's tea is nettle, not smoked lapsang."),
        _row("2030-01-05", "10:00 — Map", "Made a correction to the parish map. Alder Brannock came by."),
    ]
    for later in cases:
        res = ef.focus_candidates("What is Alder Brannock's tea?", [base, later])
        assert res.corrected_by == {}, later["text"]
    fixed = _row("2030-01-05", "10:00 — Note", "I had misheard: Alder Brannock's favourite tea is white peony.")
    assert ef.focus_candidates("What is Alder Brannock's tea?", [base, fixed]).corrected_by == {1: "2030-01-05"}
    older_fix = _row("2029-12-30", "10:00 — Correction",
                     "Correction: Alder Brannock's tea is white peony, not smoked lapsang.")
    assert ef.focus_candidates("What is Alder Brannock's tea?", [base, older_fix]).corrected_by == {}


def test_several_people_in_one_question():
    rows = [TEA[0], TEA[1],
            _row("2030-01-12", "09:00 — Bryony", "Bryony Castelow likes nettle tea."),
            _row("2030-01-13", "09:00 — Caspian", "Caspian Dunmore likes white peony.")]
    res = ef.focus_candidates("Do Alder Brannock and Bryony Castellow both like tea?", rows)
    # Brennock is confirmed someone else (Brannock is named exactly); Bryony has no exact entry: kept, hedged.
    # "Caspian Dunmore" opens its sentence, so it is not a confident person name; it scores 0.3 below the
    # best exact hit and names no asked entity, so the margin drops it.
    assert [(rank, reason) for rank, _, reason in res.removed] == [(1, "near_name"), (4, "below_margin")]
    assert [rank for rank, _ in res.kept] == [2, 3] and "may be a different person" in ef.entry_label(res, 3)
    both = ef.focus_candidates("Compare Alder Brannock and Alder Brennock.", rows)
    assert [rank for rank, _ in both.kept] == [1, 2]  # each named exactly by its own entry
    assert {reason for _, _, reason in both.removed} == {"below_margin"}


def test_name_less_follow_up_carries_the_previous_entity():
    earlier = ["What is Alder Brannock's phone number?"]
    res = ef.focus_candidates("And his email?", TEA, earlier=earlier)
    assert res.carried and _names(res.entities) == ["Alder Brannock"] and (1, TEA[0], "near_name") in res.removed
    assert "carried from the earlier question" in ef.context_notes(res)[0]
    assert not ef.focus_candidates("And his email?", TEA).active  # no earlier question: nothing to carry
    assert not ef.analyse("How has the week been?", earlier).active  # not a follow-up: carries nothing


def test_focus_limit_is_recorded_as_a_removal():
    rows = [_row("2030-01-0%d" % (i + 1), "09:00 — Tobiah", f"Tobiah visit {i}") for i in range(5)]
    res = ef.focus_candidates("What did Tobiah give me?", rows, limit=3)
    assert len(res.kept) == 3
    assert [(r, why) for r, _, why in res.removed] == [(4, "candidate_limit"), (5, "candidate_limit")]


# ---------------- the gate's near-name fixture ----------------


def test_fixture_recall_and_precision_never_fall(world):
    """Recall never falls below plain vector search; near-name distractors leave the top k. (Precision on this
    fixture barely moves: dropping only confirmed near names leaves other people's entries in the top 8.)"""
    before, after = metrics(world, focus=False), metrics(world, focus=True)
    assert before["queries"] == after["queries"] > 40
    assert after["recall"] >= before["recall"] and after["context_recall"] >= before["context_recall"]
    assert after["recall"] >= THRESHOLDS["candidate_retrieval"]["recall_at_k_min"]
    assert after["precision"] >= before["precision"]


def test_trace_accounts_for_every_vector_candidate(world):
    for q in world.queries:
        got = ask(world, q["text"], _day(q))
        focus, cands = got.trace["focus"], got.trace["candidates"]
        if not ef.analyse(q["text"]).active:
            assert focus == {"checked": False} and got.trace["params"]["candidate_k"] == 2 * 8
            continue
        assert focus["checked"] and got.trace["params"]["candidate_k"] == 4 * 8
        assert len(cands) <= 2 * 8
        assert focus["vector_candidates"] == len(cands) + len(focus["removed"])
        ranks = sorted([c["vector_rank"] for c in cands] + [r["vector_rank"] for r in focus["removed"]])
        assert ranks == list(range(1, focus["vector_candidates"] + 1))
        assert [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
        assert [c["vector_rank"] for c in cands] == sorted(c["vector_rank"] for c in cands)  # never reordered
        assert {r["id"] for r in focus["removed"]}.isdisjoint(e["id"] for e in got.trace["loaded"])
        assert got.trace["rerank"] == {"present": False}
        for r in focus["removed"]:
            assert set(r) <= {"vector_rank", "id", "day", "time", "score", "reason"}
            assert r["reason"] in {"near_name", "other_person", "below_margin", "after_time_anchor",
                                   "candidate_limit"}
        assert sum(focus["removed_by_reason"].values()) == len(focus["removed"])
        blob = json.dumps(got.trace)
        for e in ef.asked_entities(q["text"]):
            assert e.name not in blob  # counts and refs only: never the asked names
        for fact in q["expected_facts"] + q["forbidden_facts"]:
            assert fact not in blob


def test_trace_off_gives_the_same_context(world):
    ids = ("Q-needle-00", "Q-needle-10", "Q-multi-0", "Q-ambig-0", "Q-unans-4")
    picked = [world.by_id[i] for i in ids] + [q for q in world.queries if q["category"] in ("temporal", "correction")]
    for q in picked:
        asm = _assembler(world.retriever)
        assert asm.build(_day(q), q["text"]) == asm.build(_day(q), q["text"], trace=RetrievalTrace()), q["id"]


def test_a_message_without_names_or_dates_is_unchanged(world):
    for text in ("how have the survey days felt lately?", "Which film did I see at the cinema on my birthday?"):
        on, off = ask(world, text, date(2030, 3, 7)), ask(world, text, date(2030, 3, 7), focus=False)
        assert on.messages == off.messages and on.trace["candidates"] == off.trace["candidates"]
        assert on.trace["focus"] == {"checked": False} and "Asked about" not in on.block


def test_needle_context_has_no_near_name_entry(world):
    seen = 0
    for q in world.queries:
        if q["category"] not in ("needle", "paraphrase"):
            continue
        entity = ef.asked_entities(q["text"])[0]
        got, off = ask(world, q["text"], _day(q)), ask(world, q["text"], _day(q), focus=False)
        assert "near" not in _loaded_kinds(world, got, entity), q["id"]
        assert set(off.loaded) & set(q["relevant_entry_ids"][0]) <= set(got.loaded)  # recall kept
        assert all(_norm(f) in _norm(got.block) for f in q["expected_facts"]), q["id"]
        assert f'(asked about "{entity.name.lower()}". entries naming "{entity.name.lower()}"' in _norm(got.block)
        # Without the focus the asked person's near-name distractor is in the context (the #116 failure).
        assert "near" in _loaded_kinds(world, off, entity), q["id"]
        seen += 1
    assert seen >= 30


def test_followup_about_the_near_name_gets_no_conflation(world):
    seen = 0
    for q in world.queries:
        if "followup" not in q:
            continue
        f = q["followup"]
        said = q["expected_facts"][0] if q["expected_facts"] else "nothing"
        turns = [{"role": "user", "content": q["text"]}, {"role": "companion", "content": f"Your note says {said}."}]
        got = ask(world, f["text"], _day(q), turns=turns)
        entity = ef.asked_entities(f["text"])[0]
        assert score.best_recall(f["relevant_entry_ids"], got.loaded) == 1.0, q["id"]
        assert all(_norm(x) in _norm(got.block) for x in f["expected_facts"]), q["id"]
        assert "near" not in _loaded_kinds(world, got, entity), q["id"]  # the first question's person is out
        seen += 1
    assert seen >= 4


def test_correction_context_marks_only_the_superseded_entry(world):
    seen = 0
    for q in world.queries:
        if q["category"] != "correction":
            continue
        entity = ef.asked_entities(q["text"])[0]
        got = ask(world, q["text"], _day(q))
        correction = next(i for i in got.loaded if world.entries[i]["kind"] == "correction")
        cday = world.entries[correction]["day"]
        assert all(_norm(f) in _norm(got.block) for f in q["expected_facts"])
        stale = [line for line in got.block.splitlines()
                 if line.startswith("--- [") and f"correction of {cday}" in line]
        about = [e["id"] for e, kind in zip(got.trace["loaded"], _loaded_kinds(world, got, entity))
                 if kind == "match" and world.entries[world.index[(e["day"], e["time"])]]["kind"] == "needle"]
        assert len(stale) == len(about) == 1, q["id"]  # the original note only, not other people's entries
        seen += 1
    assert seen == 4


def test_time_anchored_context_leaves_out_the_later_correction(world):
    seen = 0
    for q in world.queries:
        if q["category"] != "temporal":
            continue
        got = ask(world, q["text"], _day(q))
        assert all(_norm(f) in _norm(got.block) for f in q["expected_facts"])
        assert "correction" not in _kinds(world, got.loaded), q["id"]
        assert "time anchor" in _norm(got.block)
        assert any(r["reason"] == "after_time_anchor" for r in got.trace["focus"]["removed"])
        seen += 1
    assert seen == 2


def test_deleted_entry_context_has_no_near_name_value(world):
    seen = 0
    for q in world.queries:
        if q["category"] != "deletion":
            continue
        entity = ef.asked_entities(q["text"])[0]
        got, off = ask(world, q["text"], _day(q)), ask(world, q["text"], _day(q), focus=False)
        assert q["expect_abstain"]
        assert "near" not in _loaded_kinds(world, got, entity), q["id"]
        assert any(r["reason"] == "near_name" for r in got.trace["focus"]["removed"])
        # Without the focus the near-name entry is in the context: the #116 deleted-entry failure.
        assert "near" in _loaded_kinds(world, off, entity), q["id"]
        seen += 1
    assert seen == 3


def _kinds(world, ids):
    return {world.entries[i]["kind"] if i in world.entries else i for i in ids}


# ---------------- negative cases outside the gate fixture (review findings 1-5) ----------------


def _corpus(tmp_path, entries):
    """entries: (ISO day, HH:MM, topic, text) -> a retriever over them and {chunk id: topic}."""
    by_day = {}
    for day, hhmm, topic, me in entries:
        xid = hashlib.sha1(topic.encode()).hexdigest()[:12]
        by_day.setdefault(day, []).append(gen_fixture.render_block(f"{hhmm} — {topic}", me, "Noted.", xid))
    files = {gen_fixture.day_path(date.fromisoformat(d)): gen_fixture.render_day(date.fromisoformat(d), blocks)
             for d, blocks in by_day.items()}
    r = Retriever(tmp_path / "small.db", TrigramLLM())
    _index_files(r, files)
    return r, {row[0]: row[1].split(" — ", 1)[1] for row in r._conn.execute("SELECT id, header FROM chunks")}


def _case(tmp_path, entries, question, relevant, turns=None):
    """Recall is 1.0 with the focus on, and nothing plain vector search loads is lost."""
    r, topic = _corpus(tmp_path, entries)
    on, off = RetrievalTrace(), RetrievalTrace()
    messages = _assembler(r, min_score=0.0).build(date(2030, 6, 1), question, session_turns=turns, trace=on)
    _assembler(r, focus=False, min_score=0.0).build(date(2030, 6, 1), question, session_turns=turns, trace=off)
    loaded = {topic[e["id"]] for e in on.data["loaded"]}
    assert set(relevant) <= loaded, (question, loaded)
    assert set(relevant) & {topic[e["id"]] for e in off.data["loaded"]} <= loaded
    removed = {topic[e["id"]]: e["reason"] for e in on.data["focus"].get("removed", [])}
    named = {t: why for t, why in removed.items() if why != "below_margin"}  # the margin depends on stub scores
    return SimpleNamespace(block=_retrieved_block(messages), removed=removed, named=named, trace=on.data)


BOAT = ("2030-01-01", "09:00", "Boat", "Alder Brannock showed me his boat, the Curlew.")
COFFEE = ("2030-01-03", "10:00", "Coffee", "Had coffee with Alder Brennock; his number is 0111 222.")
FILLER = ("2030-01-04", "11:00", "Studio", "Inked the legend for the coastal chart.")


def test_first_name_only_entry_is_kept(tmp_path):
    phone = ("2030-01-02", "09:10", "Phone", "Alder called, new number is 0789 552.")
    got = _case(tmp_path, [BOAT, phone, COFFEE, FILLER], "What is Alder Brannock's phone number?", ["Phone", "Boat"])
    assert got.named == {"Coffee": "near_name"}


def test_title_with_or_without_a_full_stop(tmp_path):
    knee = ("2030-01-02", "09:10", "Knee", "Saw Dr. Patel about my knee; rest it for a week.")
    clinic = ("2030-01-03", "09:10", "Clinic", "Dr Patil at the clinic said to stretch my knee.")
    got = _case(tmp_path, [knee, clinic, FILLER], "What did Dr Patel say about my knee?", ["Knee"])
    assert got.named == {"Clinic": "near_name"}


def test_typo_in_the_question_keeps_and_hedges(tmp_path):
    got = _case(tmp_path, [BOAT, COFFEE, FILLER], "What did Alder Branock show me?", ["Boat", "Coffee"])
    assert got.removed == {} and "do not answer" not in got.block.lower()
    assert got.block.count("may be a different person, check the name") == 2


def test_nickname_on_either_side_is_kept(tmp_path):
    sail = ("2030-01-02", "09:10", "Sail", "Al Brannock took me sailing on the Curlew.")
    got = _case(tmp_path, [BOAT, sail, COFFEE, FILLER], "Where did Alder Brannock take me sailing?", ["Sail", "Boat"])
    assert "Sail" not in got.removed and got.named == {"Coffee": "near_name"}
    short = _case(tmp_path / "b", [BOAT, COFFEE, FILLER], "Which boat did Al show me?", ["Boat"])
    assert short.removed == {}


def test_until_in_a_statement_is_not_a_time_anchor(tmp_path):
    back = ("2030-03-05", "08:00", "Back", "Sam got back from Lisbon today.")
    away = ("2030-02-20", "08:00", "Away", "Sam left for Lisbon this morning.")
    got = _case(tmp_path, [back, away, FILLER], "Sam is away until 3 March 2030. Is Sam back yet?", ["Back", "Away"])
    assert got.named == {} and "Time anchor" not in got.block


def test_incidental_corrected_supersedes_nothing(tmp_path):
    tea = ("2030-01-01", "09:00", "Tea", "Alder Brannock's favourite tea is smoked lapsang.")
    marking = ("2030-01-05", "10:00", "Marking", "Corrected the essays with Alder Brannock all afternoon.")
    got = _case(tmp_path, [tea, marking, FILLER], "What is Alder Brannock's favourite tea?", ["Tea"])
    assert "correction of" not in got.block and got.trace["focus"]["superseded_labelled"] == 0


def test_name_less_follow_up_carries_the_person(tmp_path):
    email = ("2030-01-02", "09:10", "Email", "Bryony Castellow's email is bryony at example dot org.")
    other = ("2030-01-03", "09:10", "Email two", "Bryony Castelow's email is castelow at example dot net.")
    phone = ("2030-01-01", "09:10", "Phone", "Bryony Castellow's phone is 0123 456.")
    turns = [{"role": "user", "content": "What is Bryony Castellow's phone number?"},
             {"role": "companion", "content": "It is 0123 456."}]
    got = _case(tmp_path, [email, other, phone, FILLER], "And her email?", ["Email"], turns=turns)
    assert got.named == {"Email two": "near_name"} and got.trace["focus"]["carried"] is True
    assert "carried from the earlier question" in got.block


def test_several_people_in_one_question(tmp_path):
    tea = ("2030-01-05", "09:00", "Nettle", "Bryony Castellow likes nettle tea.")
    near = ("2030-01-06", "09:00", "Peony", "Bryony Castelow likes white peony.")
    got = _case(tmp_path, [BOAT, COFFEE, tea, near, FILLER], "What do Alder Brannock and Bryony Castellow like?",
                ["Boat", "Nettle"])
    assert got.named == {"Coffee": "near_name", "Peony": "near_name"}


NEAR_ONLY = """## Monday, January 14, 2030

### 08:10 — Chat with Alder Brennock

**Me:** Had coffee with Alder Brennock near the boatyard. Funny coincidence: their favourite tea is nettle and mint.

**Assistant:** Marit had coffee with Alder Brennock, whose favourite tea is nettle and mint.

### 13:20 — Studio day

**Me:** Spent the session inking the legend for the coastal chart. Bright and cold.

**Assistant:** Marit spent the session inking the legend for the coastal chart.
"""


def test_entry_removed_outright_context_hedges_the_near_name(tmp_path):
    """The asked person's entry is gone entirely: nothing names them, so nothing is dropped, the near-name
    entry is labelled as possibly someone else, and the note says to check the name or abstain."""
    r = Retriever(tmp_path / "idx.db", TrigramLLM())
    _index_files(r, {"Entries/2030/January/January 14, 2030.md": NEAR_ONLY})
    question = "What is Alder Brannock's favourite tea?"
    trace = RetrievalTrace()
    block = _retrieved_block(_assembler(r).build(date(2030, 3, 7), question, trace=trace))
    assert '(Asked about "Alder Brannock". No retrieved entry names "Alder Brannock" exactly.' in block
    assert "if no entry records what was asked, say so" in block and "do not answer" not in block.lower()
    header = next(line for line in block.splitlines() if "Chat with Alder Brennock" in line)
    assert header.endswith('similar to "Alder Brannock": may be a different person, check the name) ---')
    assert trace.data["focus"]["similar_name_labelled"] == 1 and not trace.data["focus"]["applied"]
    off = RetrievalTrace()
    plain = _retrieved_block(_assembler(r, focus=False).build(date(2030, 3, 7), question, trace=off))
    assert [e["id"] for e in trace.data["loaded"]] == [e["id"] for e in off.data["loaded"]] != []
    assert "a different person" not in plain and "Asked about" not in plain


def test_notes_and_labels_are_charged_to_the_budget(world):
    q = next(x for x in world.queries if x["category"] == "correction")
    got = ask(world, q["text"], _day(q))
    budget = got.trace["budget"]
    assert budget["used_chars"] <= budget["max_context_chars"]
    assert budget["used_chars"] > sum(e["chars"] for e in got.trace["loaded"])  # the opening note is charged
    assert any("correction of" in line for line in got.block.splitlines())


def test_system_prompt_states_the_name_correction_and_abstain_rules():
    text = (Path(__file__).resolve().parent.parent / "config" / "prompts" / "system.md").read_text()
    for rule in ("Keep people apart", "Similar names: check, do not assume", "The latest record wins",
                 "Withdrawn means not recorded", "Abstain plainly", "Time-anchored questions", "Shared first names"):
        assert rule in text


@pytest.mark.parametrize("value", [False, "false", "0", "off", "no"])
def test_entity_focus_switch_turns_the_check_off(world, value):
    q = world.by_id["Q-needle-00"]
    asm = ContextAssembler(FakeStore(), world.retriever, _cfg(value))
    asm._system_prompt = lambda: "SYSTEM"
    trace = RetrievalTrace()
    asm.build(_day(q), q["text"], trace=trace)
    assert trace.data["focus"] == {"checked": False} and trace.data["params"]["candidate_k"] == 16


# ---------------- other identified people (coordinator decision on #117) ----------------

SOLO = ("2030-01-05", "09:00", "Solo", "Went sailing alone with Bryony Castellow on the estuary.")
BOTH = ("2030-01-06", "09:00", "Together", "Alder Brannock and Bryony Castellow went sailing on the estuary.")
CASPIAN = ("2030-01-07", "09:00", "Leeds", "Sailed near Leeds with Caspian Dunmore.")
WALK = ("2030-01-08", "09:00", "Walk", "Walked to Gullet Cove and back along the estuary.")


def test_other_people_only_entries_go_when_the_asked_person_is_named(tmp_path):
    got = _case(tmp_path, [BOAT, BOTH, SOLO, CASPIAN, WALK, FILLER], "Where did Alder Brannock go sailing?",
                ["Boat", "Together"])
    # The entry naming the asked person alongside another stays; unnamed and place-only entries are never
    # other_person (only the score margin can drop them).
    assert got.named == {"Solo": "other_person", "Leeds": "other_person"}
    assert got.trace["focus"]["removed_by_reason"]["other_person"] == 2
    assert "may be incomplete" in got.block


def test_no_exact_hit_drops_nothing(tmp_path):
    got = _case(tmp_path, [BOAT, BOTH, SOLO, CASPIAN, WALK], "Where did Alder Branock go sailing?",
                ["Boat", "Together", "Solo", "Leeds", "Walk"])
    assert got.removed == {}  # neither other_person nor below_margin without a full exact hit


def test_a_first_name_question_drops_no_other_people(tmp_path):
    got = _case(tmp_path, [BOAT, SOLO, CASPIAN], "Where did Alder go sailing?", ["Boat"])
    assert "other_person" not in got.removed.values()  # a single word does not identify a person for certain


def test_several_asked_people_keep_their_entries(tmp_path):
    got = _case(tmp_path, [BOAT, BOTH, SOLO, CASPIAN, WALK],
                "Where did Alder Brannock and Bryony Castellow go sailing?", ["Boat", "Together", "Solo"])
    assert got.named == {"Leeds": "other_person"}


def _rows(*texts, scores=None):
    scores = scores or [0.8] * len(texts)
    return [_row(f"2030-01-{i + 1:02d}", "09:00 — Note", t, s) for i, (t, s) in enumerate(zip(texts, scores))]


def _kept_reasons(question, rows, **kw):
    res = ef.focus_candidates(question, rows, **kw)
    return [rank for rank, _ in res.kept], {rank: why for rank, _, why in res.removed}


# Review finding 1: the asked person's own entry with a middle name, an initial, or in reverse order.
@pytest.mark.parametrize("question, text", [
    ("Where did Alder Brannock go sailing?", "Went to Leeds; Alder John Brannock sailed there."),
    ("Where did Alder Brannock go sailing?", "Went to Leeds; Alder J Brannock sailed there."),
    ("Where did Alder Brannock go sailing?", "Went to Leeds; Alder J. Brannock sailed there."),
    ("Where did Alder Brannock go sailing?", "Went to Leeds; Brannock Alder sailed there."),
    ("Where did Mary Smith go?", "Went to York with Mary Ann Smith."),
    ("Where did Alder Brannock go sailing?", "Went to Leeds; A. Brannock sailed there."),
])
def test_middle_names_initials_and_reverse_order_stay(question, text):
    exact = "Went sailing with " + ef.asked_entities(question)[0].name + " on the estuary."
    kept, removed = _kept_reasons(question, _rows(exact, text, scores=[0.9, 0.5]))
    assert kept == [1, 2] and removed == {}  # far below the margin, yet about the asked person


def test_extended_names_classify_as_about():
    assert ef.classify("Saw Alder John Brannock.", _ent("Alder Brannock"))[0] == "extended"
    assert ef.classify("Saw Brannock Alder.", _ent("Alder Brannock"))[0] == "extended"
    assert ef.classify("Saw A. Brannock.", _ent("Alder Brannock"))[0] == "variant"


# Review finding 2: an asked person with no exact hit keeps an entry sharing their surname.
@pytest.mark.parametrize("question", ["Compare Alder Brannock and Bree Castellow.",
                                      "Compare Alder Brannock and Bryn Castellow."])
def test_unmatched_asked_person_keeps_a_shared_name_entry(question):
    rows = _rows("Went sailing with Alder Brannock.", "Saw Bryony Castellow start work as a dentist.",
                 scores=[0.9, 0.5])
    assert _kept_reasons(question, rows) == ([1, 2], {})


# Review finding 3 and decision A: capitalised non-names, headers and role labels are never people or names.
@pytest.mark.parametrize("header, text", [
    ("09:00 — Note", "Visited Leeds yesterday. Felt great."),
    ("09:00 — Note", "Saw Marit at the market. She was cheerful."),
    ("09:00 — Note", "Called Mum. Felt better afterwards."),
    ("09:00 — Knee Physio", "He told me to rest it for a week."),
    ("09:00 — Sailing Trip", "Great wind today, fifteen knots."),
    ("09:00 — Note", "Morning Pages\nFelt calm after the ride."),
    ("09:00 — Note", "Coffee Time with nobody. Weekly Review done."),
    ("09:00 — Note", "Read Mr Men to the kids."),
    ("09:00 — Note", "**Bryony Castellow:** went to Leeds."),
])
def test_entries_naming_nobody_are_never_other_person(header, text):
    rows = [_row("2030-01-01", "09:00 — Boat", "Went sailing with Alder Brannock.", 0.8),
            _row("2030-01-02", header, text, 0.8)]
    assert _kept_reasons("Where did Alder Brannock go?", rows) == ([1, 2], {})
    assert ef.identified_people(ef._body({"text": f"### {header}\n{text}"})) == []


@pytest.mark.parametrize("surname", ["Merriman", "Wenlock", "Hesketh", "Annand"])
def test_role_labels_and_function_words_are_never_name_variants(surname):
    entity = _ent(f"Rosalind {surname}")
    text = "**Me:** We got talking. He said so. An odd day. In the end it rained."
    assert ef.classify(text, entity) == (None, None)
    rows = [_row("2030-01-01", "09:00 — Tea", f"Ran into Rosalind {surname} at the market.", 0.8),
            _row("2030-01-02", "09:00 — Chat", "**Me:** We got talking with Bryony Castellow at the market.", 0.8)]
    assert _kept_reasons(f"What is Rosalind {surname}'s tea?", rows) == ([1], {2: "other_person"})


def test_places_organisations_and_works_are_not_people():
    for phrase in ("Blue Heron", "Peak District", "Lake District", "New York", "Red Kite", "Hollin Moor"):
        assert not ef.is_person(_ent(phrase)), phrase
        assert ef.identified_people(f"We went past {phrase} today.") == [], phrase
    rows = _rows("Blue Heron decided to expand.", "Then Red Kite decided to shrink.")
    assert _kept_reasons("What did Blue Heron decide?", rows) == ([1, 2], {})  # not a person: no other_person
    assert ef.identified_people("Had tea with Bryony Castellow and Dr. Patel.") == [("bryony", "castellow"), ("patel",)]
    assert ef.identified_people("Met Bryony Castellow and Dr. Patel.") == [("patel",)]  # opens a sentence
    assert ef.identified_people("Bryony Castellow came round.") == []  # sentence-initial: not confident
    assert ef.identified_people("Projected it in the Eckert IV, and the NHS Trust wrote.") == []  # caps: not names


# Review finding 4: the accepted recall limits, each pinned so the behaviour is deliberate.
def test_limit_relational_question_drops_nothing_for_not_naming():
    rows = _rows("Went sailing with Alder Brannock.", "Saw Bryony Castellow start work as a dentist.",
                 "Inked the coastal chart.", scores=[0.9, 0.5, 0.5])
    res = ef.focus_candidates("What does Alder Brannock's sister do for work?", rows)
    assert res.relational and res.removed == []
    assert not ef.focus_candidates("What is Alder Brannock's favourite tea?", rows).relational


def test_limit_unrelated_nickname_beside_another_person_is_dropped():
    rows = _rows("Went to Leeds with Robert Hale.", "Went sailing; Ziggy and then Bryony Castellow came too.")
    assert _kept_reasons("Where did Robert Hale go?", rows) == ([1], {2: "other_person"})  # accepted limit


def test_lower_case_mention_of_the_asked_person_stays():
    rows = _rows("Went sailing with Alder Brannock.", "went sailing with alder brannock and then Bryony Castellow",
                 scores=[0.9, 0.5])
    assert _kept_reasons("Where did Alder Brannock go?", rows) == ([1, 2], {})


def test_limit_pronoun_entry_beyond_the_margin_is_dropped_and_the_note_says_so():
    rows = _rows("Went sailing with Alder Brannock.", "He told me to rest it for a week.", scores=[0.9, 0.8])
    res = ef.focus_candidates("What did Alder Brannock tell me?", rows)
    assert [(r, why) for r, _, why in res.removed] == [(2, "below_margin")]
    assert "may be incomplete" in ef.context_notes(res)[0] and "pronoun" in ef.context_notes(res)[0]


def test_limit_a_work_title_can_read_as_a_person():
    assert ef.identified_people("Started reading Moby Dick again.") == [("moby", "dick")]  # documented limit


# Decision B: the score margin.
def test_margin_keeps_an_unnamed_hit_within_it_and_drops_one_beyond():
    rows = _rows("Went sailing with Alder Brannock.", "Inked the coastal chart.", "Sorted the plan chest.",
                 scores=[0.80, 0.751, 0.749])
    kept, removed = _kept_reasons("Where did Alder Brannock go?", rows)
    assert ef.MARGIN == 0.05 and kept == [1, 2] and removed == {3: "below_margin"}
    assert _kept_reasons("Where did Alder Brannock go?", rows, margin=0.10) == ([1, 2, 3], {})


def test_margin_needs_a_full_exact_hit_and_never_touches_a_possible_mention():
    rows = _rows("Went sailing with Alder Brannock.", "Alder called about the boat.", "Al came round.",
                 "Had tea with Alder Brennock.", "Inked the coastal chart.", scores=[0.9, 0.3, 0.3, 0.3, 0.3])
    kept, removed = _kept_reasons("Compare Alder Brannock and Tobiah Wren.", rows)
    # Tobiah Wren has no full exact hit, so the margin stays off (its entries may use a form not seen).
    assert kept == [1, 2, 3, 5] and removed == {4: "near_name"}
    kept, removed = _kept_reasons("Where did Alder Brannock go?", rows)
    assert kept == [1, 2, 3] and removed == {4: "near_name", 5: "below_margin"}
    assert _kept_reasons("Where did Alder Branock go?", rows) == ([1, 2, 3, 4, 5], {})  # no exact hit


def test_margin_applies_to_a_non_person_match():
    rows = _rows("Day 1 of the Hollin Moor survey at Scarrow.", "Day 2 of the Beckside survey at Lowfold.",
                 scores=[0.9, 0.7])
    question = "Which places did I survey during the Hollin Moor survey?"
    assert _kept_reasons(question, rows) == ([1], {2: "below_margin"})


def test_margin_drops_are_counted_on_their_own(world):
    q = world.by_id["Q-needle-00"]
    got = ask(world, q["text"], _day(q))
    counts = got.trace["focus"]["removed_by_reason"]
    assert counts.get("below_margin", 0) > 0 and sum(counts.values()) == len(got.trace["focus"]["removed"])
    assert got.trace["focus"]["margin"] == ef.MARGIN
    cfg = _cfg().as_dict()
    cfg["retrieval"] = {**cfg["retrieval"], "entity_focus_margin": 1.0}
    asm = ContextAssembler(FakeStore(), world.retriever, Config(cfg))
    asm._system_prompt = lambda: "SYSTEM"
    wide = RetrievalTrace()
    asm.build(_day(q), q["text"], trace=wide)
    assert "below_margin" not in wide.data["focus"]["removed_by_reason"] and wide.data["focus"]["margin"] == 1.0


def test_people_and_places_are_told_apart():
    found = ef.identified_people("Had coffee with Bryony Castellow and Dr. Patel.")
    assert found == [("bryony", "castellow"), ("patel",)]
    assert ef.identified_people("Marit walked to Gullet Cove, then the Hollin Moor survey.") == []
    assert ef.identified_people("**Me:** Spent the session inking.") == []
    assert ef.is_person(_ent("Alder Brannock")) and ef.is_person(_ent("Dr Patel"))
    assert not ef.is_person(_ent("Tobiah")) and not ef.is_person(_ent("Hollin Moor"))


# The independent review's probes of af2730b (r117_probe*.py), kept as regressions: every hit scores 0.5,
# so only the name rules act. Expected: {rank: reason} of the removed hits.
_A = "Alder Brannock went sailing on the estuary."
_BC = "Bryony Castellow started work as a dentist."
_NONE = "Walked along the estuary with the dog."


@pytest.mark.parametrize("question, texts, removed", [
    ("What does Alder Brannock's sister do for work?", [_A, _BC, _NONE], {}),
    ("Compare Alder Brannock and Bree Castellow.", [_A, _BC, _NONE], {}),
    ("Where did Alder Brannock go?", [_A, "went sailing with alder brannock and Bryony Castellow"], {}),
    ("Where did Alder Brannock go?", [_A, "Alder and Bryony Castellow went to Leeds."], {}),
    ("Where did Alder Brannock go?", [_A, "Al and Bryony Castellow went to Leeds."], {}),
    ("Where did Alder Brannock go?", [_A, "Bryony Brannock went to Leeds."], {}),
    ("Where did Zoë Brännock go?", ["Zoe Brannock's boat sank.", _BC], {}),
    ("What did Ann Lee say?", ["Ann Lee said hello.", "Joanne Lee said goodbye."], {}),
    ("Where did Alder Brannok go?", [_A, _BC], {}),
    ("Where did Alder Brannock go?", [_A, "The Brannocks visited Leeds."], {}),
    ("What did Alder Brannock say?", [_A, "Then Dr Patel said my knee is fine."], {2: "other_person"}),
    ("What did Alder Brannock say?", [_A, "Mary-Jane Watson said hi."], {}),
    ("Where did A. Brannock go?", [_A, _BC], {}),
    ("What did Dr Patel say?", ["Dr. Patel said rest.", "Then Dr Chen said exercise.", "Raj Patel said hi"],
     {2: "other_person"}),
    ("Where did Alder Brannock go?", [_A, "Then Aunt Peg phoned."], {2: "other_person"}),
    ("Where did Alder Brannock go?", [_A, "**Bryony Castellow:** went to Leeds."], {}),
    ("What is Tobiah Wren's number?", ["Alder Brannock number is 1", "Coffee at the market."], {}),
    ("What is Alder Brannock's number?", ["Alder Brannock number is 1", "Brannock called, new number 2.",
                                          "Alder Brennock number is 3"], {3: "near_name"}),
])
def test_review_probes(question, texts, removed):
    assert _kept_reasons(question, _rows(*texts, scores=[0.5] * len(texts)))[1] == removed


def test_fixture_never_drops_an_entry_about_an_asked_entity(world):
    """Over every fixture query: no removed hit names an asked entity in any about-form, and every
    other_person drop names, in its body, a person the heuristic is confident about."""
    for q in world.queries:
        got = ask(world, q["text"], _day(q))
        entities = ef.asked_entities(q["text"])
        for r in got.trace["focus"].get("removed", []):
            if r["reason"] in ("after_time_anchor", "candidate_limit"):
                continue
            text = _text(world.retriever, r["id"])
            assert not any(ef.classify(text, e)[0] in ef.ABOUT for e in entities), (q["id"], r)
            if r["reason"] == "other_person":
                assert ef.identified_people(ef._body({"text": text})), (q["id"], r)


# ---------------- re-review of d9238ff (findings 1-8), probes /tmp/rr117-probes kept as regressions ----------------


@pytest.mark.parametrize("question, texts", [
    ("What did I do at Christmas?", ["Christmas with the family, lovely dinner.",
                                     "Boxing day: long walk after the big family dinner.",
                                     "Wrapped presents, dinner prep, lights on the tree."]),
    ("How was my trip to Paris?", ["Arrived in Paris, hotel is small.", "Day two: the Louvre, then a long lunch."]),
    ("What did I learn about Python?", ["Python decorators finally clicked.",
                                        "Spent the evening on generators and asyncio."]),
])
def test_a_lone_topic_word_never_arms_the_margin(question, texts):
    scores = [0.9] + [0.6] * (len(texts) - 1)
    assert _kept_reasons(question, _rows(*texts, scores=scores)) == (list(range(1, len(texts) + 1)), {})


def test_multi_word_and_titled_names_arm_the_margin():
    for question, exact in [("Where did Alder Brannock go?", "Went sailing with Alder Brannock."),
                            ("What did Dr Patel say?", "Dr Patel said rest."),
                            ("Which places did I survey during the Hollin Moor survey?", "Day 1 of the Hollin Moor survey.")]:
        assert _kept_reasons(question, _rows(exact, "Inked the coastal chart.", scores=[0.9, 0.6]))[1] == {2: "below_margin"}


@pytest.mark.parametrize("best, score, dropped", [
    (0.8, 0.75, False), (0.9, 0.85, False), (0.75, 0.70, False), (0.67, 0.62, False), (0.55, 0.50, False),
    (0.7, 0.65, False), (0.704, 0.654, False), (0.8, 0.7499, True), (0.8, 0.7501, False), (0.9, 0.8499, True),
])
def test_margin_boundary_is_exact(best, score, dropped):
    rows = _rows("Went sailing with Alder Brannock.", "Walked along the estuary with the dog.", scores=[best, score])
    assert (2 in _kept_reasons("Where did Alder Brannock go?", rows)[1]) is dropped


@pytest.mark.parametrize("value, expected", [
    (0, 0.0), (0.1, 0.1), ("0.1", 0.1), (1, 1.0), (-1, ef.MARGIN), (-0.2, ef.MARGIN), (1.5, ef.MARGIN),
    ("abc", ef.MARGIN), ("", ef.MARGIN), ("0,05", ef.MARGIN), (None, ef.MARGIN), (True, ef.MARGIN),
    ("nan", ef.MARGIN), (float("nan"), ef.MARGIN), (float("inf"), ef.MARGIN),
])
def test_margin_setting_is_validated(value, expected):
    assert ef.parse_margin(value) == expected


@pytest.mark.parametrize("value", [None, "abc", "", -1, 1.5, "nan", float("nan"), float("inf"), True])
def test_bad_margin_setting_never_breaks_the_chat_or_the_trace(tmp_path, value, caplog):
    r = Retriever(tmp_path / "m.db", TrigramLLM())
    _index_files(r, {"Entries/2030/January/January 1, 2030.md": gen_fixture.render_day(date(2030, 1, 1), [
        gen_fixture.render_block("09:00 — Boat", "Alder Brannock went sailing on the estuary.", "Noted.", "a1"),
        gen_fixture.render_block("10:00 — Walk", "Walked along the estuary with the dog.", "Noted.", "a2")])})
    cfg = _cfg().as_dict()
    cfg["retrieval"] = {**cfg["retrieval"], "entity_focus_margin": value}
    asm = ContextAssembler(FakeStore(), r, Config(cfg))
    asm._system_prompt = lambda: "SYSTEM"
    trace = RetrievalTrace()
    asm.build(date(2030, 2, 1), "Where did Alder Brannock go sailing?", trace=trace)
    assert trace.data["focus"]["margin"] == ef.MARGIN
    json.dumps(trace.data, allow_nan=False)  # strict JSON: no nan or inf


def test_margin_passed_directly_is_validated_too():
    rows = _rows("Went sailing with Alder Brannock.", "Walked along the estuary with the dog.", scores=[0.8, 0.9])
    for bad in (-0.1, float("nan"), 2):
        res = ef.focus_candidates("Where did Alder Brannock go?", rows, margin=bad)
        assert res.margin == ef.MARGIN and res.removed == []  # a hit above the best is never "below"


def test_config_yaml_documents_the_margin():
    text = (Path(__file__).resolve().parent.parent / "config" / "config.yaml").read_text()
    assert "entity_focus_margin: 0.05" in text


@pytest.mark.parametrize("text", [
    "- Visited Leeds yesterday.", "* Called Mum after lunch.", "1. Saw Marit at the market.", "1) Called Mum",
    "> Saw Marit at the market", "“Called Mum yesterday,” I said.", "He said: Called Mum",
    "— Visited Leeds", "• Visited Leeds", "Went out.\n\n- Met Marit\n- Phoned Mum\n- Cleaned Kitchen",
    "Then: Visited Leeds", "Done) Called Mum", "Today… Called Mum", "- **Visited Leeds** today",
    "(Visited Leeds) fine", "Went to the pub. Then Visited Leeds again", "Nice day. Mum Visited Leeds",
    "TODO: Call Mum", "Spent the day. Next Steps: Call Mum", "I think Mum Called yesterday",
    "Went out – Visited Leeds after", "[Visited Leeds] fine",
])
def test_list_markers_quotes_and_dashes_open_a_sentence(text):
    rows = [_row("2030-01-01", "09:00 — Boat", _A, 0.5), _row("2030-01-02", "09:00 — Note", text, 0.5)]
    assert _kept_reasons("Where did Alder Brannock go?", rows) == ([1, 2], {})
    assert ef.identified_people(text) == []


def test_titled_people_after_an_opening_word_still_count():
    assert ef.identified_people("Then Dr Patel said rest.") == [("patel",)]
    assert ef.identified_people("Then Aunt Peg phoned.") == [("peg",)]
    assert ef.identified_people("Called Mum, then Aunt Peg.") == [("peg",)]


def test_unmatched_asked_person_disarms_the_margin():
    rows = _rows("Alder Brannock went sailing.", "Bryony called about the dentist job.", scores=[0.9, 0.8])
    assert _kept_reasons("Compare Alder Brannock and Bree Castellow.", rows) == ([1, 2], {})
    rows = _rows("Alder Brannock went sailing.", "Spent the day on the dentist job.", scores=[0.9, 0.7])
    assert _kept_reasons("Compare Alder Brannock and Bree Castellow.", rows) == ([1, 2], {})
    both = _rows("Alder Brannock went sailing.", "Bryony Castellow started at the dentist.", "Inked the chart.",
                 scores=[0.9, 0.85, 0.6])
    assert _kept_reasons("Compare Alder Brannock and Bryony Castellow.", both)[1] == {3: "below_margin"}


@pytest.mark.parametrize("asked, ascii_text", [
    ("Søren Kierkegaard", "Spoke with Soren Kierkegaard."), ("Łukasz Nowak", "Met Lukasz Nowak."),
    ("Hans Strauß", "Met Hans Strauss."), ("Hans Strauß", "Met Hans STRAUSS."), ("Zoë Brännock", "Met Zoe Brannock."),
    ("Ægir Dahl", "Met Aegir Dahl."), ("Đorđe Ilić", "Met Djordje Ilic."),
])
def test_letters_without_a_decomposition_fold_for_matching(asked, ascii_text):
    rows = _rows(f"Met {asked}.", ascii_text, scores=[0.9, 0.85])
    kept, removed = _kept_reasons(f"What did {asked} say?", rows)
    if asked.startswith("Đ"):  # "Dj" is a transliteration, not a fold: a near spelling, as for Mueller
        assert removed in ({}, {2: "near_name"})
    else:
        assert kept == [1, 2] and removed == {}
    assert ef._key("Søren") == "soren" and ef._key("Łukasz") == "lukasz" and ef._key("Strauß") == "strauss"


def test_lower_case_mention_of_a_one_word_name_is_protected():
    rows = _rows("Saw Marit at the market.", "saw marit again. said it would rain", "Inked the chart.",
                 scores=[0.9, 0.4, 0.4])
    assert ef.could_refer([(None, None)], [_ent("Marit")], [], "saw marit again")
    assert not ef.could_refer([(None, None)], [_ent("Al")], [], "al fresco lunch")  # under three letters: no
    res = ef.focus_candidates("What did Marit say?", rows)
    assert 2 in [rank for rank, _ in res.kept]


@pytest.mark.parametrize("question, relational", [
    ("What does Alder Brannock's sister-in-law do?", True), ("What does Alder Brannock's mother-in-law think?", True),
    ("What does the sister of Alder Brannock do?", True), ("What does the wife of Alder Brannock do?", True),
    ("What does the sister-in-law of Alder Brannock do?", True),
    ("What does Alder Brannock's older sister do?", True), ("Does Alder Brannock's wife work?", True),
    ("What does Alder Brannock’s sister do?", True), ("What is Alder Brannock's favourite tea?", False),
    ("What is the boat of Alder Brannock called?", False),
])
def test_relational_questions_include_hyphenated_and_of_forms(question, relational):
    rows = _rows("Went sailing with Alder Brannock.", "Then Dr Patel started work as a dentist.", scores=[0.9, 0.8])
    res = ef.focus_candidates(question, rows)
    assert res.relational is relational
    if relational:
        assert res.removed == []


# Probes p4 and p6 of the re-review: the asked person's own entries in many spellings all stay.
@pytest.mark.parametrize("question, exact, other", [
    ("Where did Alder Brannock go?", "Alder Brannock went sailing.", "Took Alder Brannock’s boat out."),
    ("Where did Alder Brannock go?", "Alder Brannock went sailing.", "Saw A Brannock out."),
    ("Where did Alder Brannock go?", "Alder Brannock went sailing.", "Went out with Alder\nBrannock today."),
    ("Where did Alder Brannock go?", "Alder Brannock went sailing.", "Went out with Alder  Brannock today."),
    ("Where did Alder Brannock go?", "Alder Brannock went sailing.", "ALDER BRANNOCK called"),
    ("Where did Alder Brannock go?", "Alder Brannock went sailing.", "went out with alder brannock today"),
    ("Where did Sean O'Neill go?", "Sean O'Neill went home.", "Took Sean O’Neill’s boat out."),
    ("Where did Sean O'Neill go?", "Sean O'Neill went home.", "Met Sean ONeill at the pub."),
    ("Where did Mary Smith-Jones go?", "Mary Smith-Jones went home.", "Met Mary Smith Jones."),
    ("Where did Mary Smith-Jones go?", "Mary Smith-Jones went home.", "Mrs Smith-Jones called."),
    ("Where did Jean-Luc Picard go?", "Jean-Luc Picard went home.", "Jean Luc Picard came by."),
    ("What did Dr Patel say?", "Dr Patel said rest.", "Patel rang about the results."),
    ("Что сказал Иван Петров?", "Встретил Иван Петров.", "Иван Петров позвонил."),
])
def test_rereview_probe_target_spellings_stay(question, exact, other):
    kept, removed = _kept_reasons(question, _rows(exact, other, scores=[0.9, 0.3]))
    assert kept == [1, 2] and removed == {}


def test_mc_and_mac_spellings_are_near_names():
    """Accepted, like "Mueller" for "Müller": a different spelling is a near name when the exact one exists."""
    rows = _rows("Tom McDonald said hi.", "Tom MacDonald rang.", scores=[0.9, 0.3])
    assert _kept_reasons("What did Tom McDonald say?", rows) == ([1], {2: "near_name"})


# ---------------- services#118: short forms, places and brands, sentence-initial verbs ----------------

def test_listed_short_form_is_kept_beside_another_person():
    """#118 reverses the #117 limit for listed short forms: "Bob" is a short form of "Robert"."""
    rows = _rows("Went to Leeds with Robert Hale.", "Went sailing; Bob and then Bryony Castellow came too.")
    assert _kept_reasons("Where did Robert Hale go?", rows) == ([1, 2], {})


@pytest.mark.parametrize("short, full", [("Will", "William"), ("Bill", "William"), ("Jan", "Janet"),
                                         ("Peggy", "Margaret"), ("Jim", "James"), ("Liz", "Elizabeth")])
def test_short_form_in_an_entry_is_a_variant_of_the_full_name(short, full):
    entity = ef.Entity(f"{full} Ashcombe", (full.lower(), "ashcombe"))
    assert ef.classify(f"Had lunch and then {short} came over.", entity)[0] == "variant"
    assert ef.classify(f"{short} called about the boat.", entity)[0] == "variant"  # sentence-initial, a name
    rows = _rows(f"Went to Leeds with {full} Ashcombe.", f"That evening {short} sailed with Bryony Castellow.")
    assert _kept_reasons(f"Where did {full} Ashcombe go?", rows) == ([1, 2], {})


@pytest.mark.parametrize("question", ["What did Will say about the boat?", "Where did Jan go sailing?",
                                      "Is Bill still at the harbour?"])
def test_short_form_in_the_question_is_an_asked_name_matching_full_name_entries(question):
    entities = ef.asked_entities(question)
    assert len(entities) == 1
    rows = _rows("William Ashcombe and Janet Ashcombe went sailing.", "Bought bread.")
    assert ef.classify(rows[0]["text"], entities[0])[0] == "variant"
    kept, removed = _kept_reasons(question, rows)
    assert 1 in kept and removed == {}


@pytest.mark.parametrize("question", ["Will you remind me what I ate?", "May I see my notes?",
                                      "Mark the date: what did I plan?", "Bill me later, what was the cost?",
                                      "Will it rain at the harbour?", "What did I do in Jan 2030?",
                                      "What happened on 3 Jan 2030?", "Will?"])
def test_sentence_initial_modal_or_verb_and_months_are_not_asked_names(question):
    assert ef.asked_entities(question) == []


@pytest.mark.parametrize("text", ["Will you call me back?", "Mark the date in the planner.", "Bill me for it.",
                                  "May I come too?", "Will be late tonight.", "Met them in Jan 2030."])
def test_sentence_initial_verb_in_an_entry_is_not_a_short_form(text):
    for full in ("William", "Marcus", "Mary", "Janet"):
        entity = ef.Entity(f"{full} Ashcombe", (full.lower(), "ashcombe"))
        assert ef.classify(text, entity)[0] is None, (text, full)
    assert ef.identified_people(text) == []


def test_sentence_initial_verb_does_not_shield_an_other_person_entry():
    rows = _rows("Went to Leeds with William Ashcombe.", "Will you ask Bryony Castellow about it?")
    assert _kept_reasons("Where did William Ashcombe go?", rows) == ([1], {2: "other_person"})


@pytest.mark.parametrize("phrase", ["San Francisco", "Hong Kong", "Kuala Lumpur", "Orbel Maps",
                                    "Harrow Bank", "Dunmore Airport", "Kestrel Inc", "Tollan University",
                                    "Quarry Street", "Fenwick Labs", "Santa Rosa", "Los Alamos"])
def test_places_and_brands_are_not_other_people(phrase):
    assert ef.identified_people(f"Spent the afternoon with friends and then {phrase} was great.") == [], phrase
    rows = _rows("Went to Leeds with Alder Brannock.", f"The trip went well and {phrase} was busy.")
    assert _kept_reasons("Where did Alder Brannock go?", rows) == ([1, 2], {}), phrase


def test_a_listed_place_word_after_in_is_still_not_a_person():
    assert ef.identified_people("We stayed in Port Ellery for a week.") == []  # "Port" opens a place


# Review of #119: a preceding word never shields a given name and surname (each failed on f814d1a).
@pytest.mark.parametrize("text", ["Met near Daniel Ostrow by the pier.", "Walked around Marcus Fenn for hours.",
                                  "Sent it via Daniel Ostrow.", "Saw a film; visited Anna Pirelli after.",
                                  "Sat in Anna Pirelli's kitchen.", "We visited Wren Hollin twice.",
                                  "Found the cafe via Orbel Mapper."])
def test_a_person_after_a_place_word_is_still_another_person(text):
    rows = _rows("Went to Leeds with Bob Hale.", text)
    assert _kept_reasons("Where did Bob Hale go?", rows) == ([1], {2: "other_person"}), text


# Review of #119: given names, particles and surnames that are also place words still name people.
@pytest.mark.parametrize("name", ["Santa Ramirez", "Le Minh", "Rio Ferdinand", "Sri Patel", "Wei Kong",
                                  "Tom Post", "Hal Ridge", "Jo Tower", "Abu Karim", "Costa Varga",
                                  "Lucia Plaza", "Paolo Arena", "Ines Mall", "Rosa Terminal"])
def test_people_with_place_like_names_are_still_other_people(name):
    rows = _rows("Went to Leeds with Bob Hale.", f"Later we had tea with {name} at the harbour.")
    assert _kept_reasons("Where did Bob Hale go?", rows) == ([1], {2: "other_person"}), name


def test_mark_is_not_a_short_form_of_marcus():
    rows = _rows("Went to Leeds with Marcus Hale.", "Later Mark Hale and Bryony Castellow sailed.")
    assert _kept_reasons("Where did Marcus Hale go?", rows) == ([1], {2: "other_person"})
    assert "mark" not in ef.SHORT_FORMS and "sandy" not in ef.SHORT_FORMS
    assert ef.SHORT_FORMS["jan"] == ("janet",)


def test_a_real_other_person_is_still_dropped_beside_places():
    rows = _rows("Went to Leeds with Alder Brannock.", "Had tea with Bryony Castellow in Hong Kong.")
    assert ef.identified_people(rows[1]["text"]) == [("bryony", "castellow")]
    assert _kept_reasons("Where did Alder Brannock go?", rows) == ([1], {2: "other_person"})


def test_docs_describe_near_spellings_the_same_way():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    for text in (ef.__doc__, readme):
        assert "A near spelling is kept (and labelled) while no hit names that entity in full" in \
            " ".join(text.split()), text[:40]


@pytest.mark.parametrize("name", ["Lucia Plaza", "Paolo Arena"])
def test_plaza_and_arena_surnames_are_people(name):
    rows = _rows("Went to Leeds with Alder Brannock.", f"Had tea with {name}.")
    assert ef.identified_people(rows[1]["text"]) == [tuple(name.lower().split())]
    assert _kept_reasons("Where did Alder Brannock go?", rows) == ([1], {2: "other_person"})
