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
    return Config({
        "context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
        "retrieval": {"top_k": 8, "min_score": min_score, "max_context_tokens": 5200, "entity_focus": focus},
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
    return {"queries": len(rec), "recall": score.mean(rec), "precision": score.mean(prec), "context_recall": score.mean(ctx)}


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
    assert ef.asked_entities("Did Dr Patel or Dr. Patel call?")[0].keys == ("patel",) and len(ef.asked_entities("Did Dr Patel or Dr. Patel call?")) == 1


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
    _row("2030-01-09", "19:00 — Studio day", "Inked the coastal chart.", 0.7),
    _row("2030-01-30", "22:00 — Correction", "Correction to my note: Alder Brannock's tea is white peony, not smoked lapsang.", 0.6),
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
    assert ef.entry_label(res, 1) == '; names "Alder Brennock", similar to "Alder Brannock": may be a different person, check the name'
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
    older_fix = _row("2029-12-30", "10:00 — Correction", "Correction: Alder Brannock's tea is white peony, not smoked lapsang.")
    assert ef.focus_candidates("What is Alder Brannock's tea?", [base, older_fix]).corrected_by == {}


def test_several_people_in_one_question():
    rows = [TEA[0], TEA[1],
            _row("2030-01-12", "09:00 — Bryony", "Bryony Castelow likes nettle tea."),
            _row("2030-01-13", "09:00 — Caspian", "Caspian Dunmore likes white peony.")]
    res = ef.focus_candidates("Do Alder Brannock and Bryony Castellow both like tea?", rows)
    # Brennock is confirmed someone else (Brannock is named exactly); Bryony has no exact entry: kept, hedged.
    assert [(rank, reason) for rank, _, reason in res.removed] == [(1, "near_name")]
    assert [rank for rank, _ in res.kept] == [2, 3, 4] and "may be a different person" in ef.entry_label(res, 3)
    both = ef.focus_candidates("Compare Alder Brannock and Alder Brennock.", rows)
    assert both.removed == []  # each named exactly by its own entry


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
    assert len(res.kept) == 3 and [(r, why) for r, _, why in res.removed] == [(4, "candidate_limit"), (5, "candidate_limit")]


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
            assert r["reason"] in {"near_name", "after_time_anchor", "candidate_limit"}
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
        stale = [line for line in got.block.splitlines() if line.startswith("--- [") and f"correction of {cday}" in line]
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
        by_day.setdefault(day, []).append(gen_fixture.render_block(f"{hhmm} — {topic}", me, "Noted.", hashlib.sha1(topic.encode()).hexdigest()[:12]))
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
    return SimpleNamespace(block=_retrieved_block(messages), removed=removed, trace=on.data)


BOAT = ("2030-01-01", "09:00", "Boat", "Alder Brannock showed me his boat, the Curlew.")
COFFEE = ("2030-01-03", "10:00", "Coffee", "Had coffee with Alder Brennock; his number is 0111 222.")
FILLER = ("2030-01-04", "11:00", "Studio", "Inked the legend for the coastal chart.")


def test_first_name_only_entry_is_kept(tmp_path):
    phone = ("2030-01-02", "09:10", "Phone", "Alder called, new number is 0789 552.")
    got = _case(tmp_path, [BOAT, phone, COFFEE, FILLER], "What is Alder Brannock's phone number?", ["Phone", "Boat"])
    assert got.removed == {"Coffee": "near_name"}


def test_title_with_or_without_a_full_stop(tmp_path):
    knee = ("2030-01-02", "09:10", "Knee", "Saw Dr. Patel about my knee; rest it for a week.")
    clinic = ("2030-01-03", "09:10", "Clinic", "Dr Patil at the clinic said to stretch my knee.")
    got = _case(tmp_path, [knee, clinic, FILLER], "What did Dr Patel say about my knee?", ["Knee"])
    assert got.removed == {"Clinic": "near_name"}


def test_typo_in_the_question_keeps_and_hedges(tmp_path):
    got = _case(tmp_path, [BOAT, COFFEE, FILLER], "What did Alder Branock show me?", ["Boat", "Coffee"])
    assert got.removed == {} and "do not answer" not in got.block.lower()
    assert got.block.count("may be a different person, check the name") == 2


def test_nickname_on_either_side_is_kept(tmp_path):
    sail = ("2030-01-02", "09:10", "Sail", "Al Brannock took me sailing on the Curlew.")
    got = _case(tmp_path, [BOAT, sail, COFFEE, FILLER], "Where did Alder Brannock take me sailing?", ["Sail", "Boat"])
    assert "Sail" not in got.removed and got.removed == {"Coffee": "near_name"}
    short = _case(tmp_path / "b", [BOAT, COFFEE, FILLER], "Which boat did Al show me?", ["Boat"])
    assert short.removed == {}


def test_until_in_a_statement_is_not_a_time_anchor(tmp_path):
    back = ("2030-03-05", "08:00", "Back", "Sam got back from Lisbon today.")
    away = ("2030-02-20", "08:00", "Away", "Sam left for Lisbon this morning.")
    got = _case(tmp_path, [back, away, FILLER], "Sam is away until 3 March 2030. Is Sam back yet?", ["Back", "Away"])
    assert got.removed == {} and "Time anchor" not in got.block


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
    assert got.removed == {"Email two": "near_name"} and got.trace["focus"]["carried"] is True
    assert "carried from the earlier question" in got.block


def test_several_people_in_one_question(tmp_path):
    tea = ("2030-01-05", "09:00", "Nettle", "Bryony Castellow likes nettle tea.")
    near = ("2030-01-06", "09:00", "Peony", "Bryony Castelow likes white peony.")
    got = _case(tmp_path, [BOAT, COFFEE, tea, near, FILLER], "What do Alder Brannock and Bryony Castellow like?",
                ["Boat", "Nettle"])
    assert got.removed == {"Coffee": "near_name", "Peony": "near_name"}


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
