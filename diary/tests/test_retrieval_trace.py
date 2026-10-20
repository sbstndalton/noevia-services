"""Opt-in retrieval tracing (agent/retrieval_trace.py) for the synthetic Diary gate.

Synthetic entries, tenants and a hash-based stub embedder only: no real Diary, model or storage.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import agent.app as appmod
from agent.config import Config
from agent.context import ContextAssembler
from agent.retrieval import Retriever
from agent.retrieval_trace import ENV_FLAG, HEADER, RetrievalTrace
from agent.util import estimate_tokens
from tests.test_retrieval_context import FakeStore, StubLLM

A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"

# Distinctive tokens that must never appear in a trace.
WORDS_A = ["ZEBRAQUILL", "MARMOTLANTERN", "OSPREYKETTLE"]
WORDS_B = ["WALRUSTHIMBLE", "HERONCOMPASS", "LYNXBALLAST"]


def _month(words, month="2026-08"):
    parts = []
    for n, w in enumerate(words, 1):
        parts.append(f"## Saturday, August {n}, 2026\n\n### 0{n}:15 — {w} topic\n\n"
                     f"**Me:** Synthetic body {w} entry {n}.\n\n**Claude:** Synthetic reply about {w}.\n")
    return "\n".join(parts)


def _retriever(tmp_path, name, words):
    r = Retriever(tmp_path / f"{name}.db", StubLLM())
    r.reindex_file("2026-08.md", _month(words))
    if not r.vec_available:
        pytest.skip("sqlite-vec not available in this environment")
    return r


def _cfg(**retrieval):
    return Config({
        "context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
        "retrieval": {"top_k": 8, "min_score": 0.30, "max_context_tokens": 5200, **retrieval},
        "prompts": {"system_path": ""},
    })


def _assembler(retriever, **retrieval):
    asm = ContextAssembler(FakeStore(), retriever, _cfg(**retrieval))
    asm._system_prompt = lambda: "SYSTEM"
    return asm


def _row_text(retriever, chunk_id):
    return retriever._conn.execute("SELECT body FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]


# ---------------- assembler level ----------------


def test_trace_is_pure_observation_context_is_identical(tmp_path):
    asm = _assembler(_retriever(tmp_path, "a", WORDS_A))
    off = asm.build(date(2026, 9, 1), "tell me about the entries")
    trace = RetrievalTrace()
    on = asm.build(date(2026, 9, 1), "tell me about the entries", trace=trace)
    assert on == off
    assert trace.data["status"] == "ok"


def test_search_results_do_not_change_without_ids(tmp_path):
    r = _retriever(tmp_path, "a", WORDS_A)
    plain = r.search("entries", top_k=5, min_score=0.0)
    ided = r.search("entries", top_k=5, min_score=0.0, with_ids=True)
    assert all("id" not in row for row in plain)
    assert [{k: v for k, v in row.items() if k != "id"} for row in ided] == plain


def test_trace_matches_the_ids_actually_loaded(tmp_path):
    r = _retriever(tmp_path, "a", WORDS_A)
    asm = _assembler(r)
    trace = RetrievalTrace()
    messages = asm.build(date(2026, 9, 1), "tell me about the entries", trace=trace)
    data = trace.data
    context = messages[1]["content"]

    loaded_ids = [e["id"] for e in data["loaded"]]
    assert len(loaded_ids) == 3 and data["counts"]["loaded"] == 3
    for e in data["loaded"]:
        body = _row_text(r, e["id"])
        assert body in context  # that chunk really is in the model's context
        assert e["tokens"] == estimate_tokens(body)
        assert e["chars"] > len(body)  # charged against the budget: text + header + framing
    # candidates are in score order and the loaded ones are exactly the flagged ones
    scores = [c["score"] for c in data["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert [c["id"] for c in data["candidates"] if c["loaded"]] == loaded_ids
    assert data["params"] == {"top_k": 8, "candidate_k": 16, "min_score": 0.30}
    assert data["rerank"] == {"present": False}
    assert data["budget"]["max_context_tokens"] == 5200 and isinstance(data["budget"]["max_context_tokens"], int)
    assert data["budget"]["used_tokens"] == sum(e["tokens"] for e in data["loaded"])
    assert data["direct_fallback"] == {"used": False, "found": False, "tokens": 0}


def test_trace_marks_top_k_and_budget_drops_and_loaded_matches_context(tmp_path):
    r = _retriever(tmp_path, "a", WORDS_A + ["PELICANRIVET", "BADGERMOSAIC"])
    # top_k=2 asks for 2*2 candidates: 4 candidates, 2 loaded, 2 dropped by top_k
    trace = RetrievalTrace()
    messages = _assembler(r, top_k=2).build(date(2026, 9, 1), "entries", trace=trace)
    assert trace.data["counts"] == {"candidates": 4, "loaded": 2, "dropped_budget": 0, "dropped_top_k": 2}
    context = messages[1]["content"]
    for c in trace.data["candidates"]:
        assert (_row_text(r, c["id"]) in context) == c["loaded"]
    # a tiny budget drops candidates by budget and loads none of them
    small = RetrievalTrace()
    msgs = _assembler(r, max_context_tokens=1).build(date(2026, 9, 1), "entries", trace=small)
    assert small.data["counts"]["loaded"] == 0 and small.data["counts"]["dropped_budget"] == 5
    assert all(c["dropped"] == "budget" for c in small.data["candidates"])
    assert "(no older entries matched this message)" in msgs[1]["content"]


def test_trace_contains_no_entry_text_header_query_or_embedding(tmp_path):
    r = _retriever(tmp_path, "a", WORDS_A)
    trace = RetrievalTrace()
    query = "SECRETQUERYPHRASE about the entries"
    _assembler(r).build(date(2026, 9, 1), query, session_turns=[
        {"role": "user", "content": "EARLIERTURNPHRASE"}, {"role": "companion", "content": "ok"}], trace=trace)
    blob = json.dumps(trace.data)
    for needle in WORDS_A + ["SECRETQUERYPHRASE", "EARLIERTURNPHRASE", "Synthetic body", "topic", "SYSTEM", "embedding"]:
        assert needle not in blob
    allowed = {"id", "rank", "day", "time", "score", "loaded", "dropped", "tokens", "chars"}
    for row in trace.data["candidates"] + trace.data["loaded"]:
        assert set(row) <= allowed
        assert all(isinstance(v, (int, float, str, bool, type(None))) for v in row.values())


def test_trace_status_no_retriever_and_search_failed():
    t = RetrievalTrace()
    _assembler(None).build(date(2026, 9, 1), "hello", trace=t)
    assert t.data["status"] == "no_retriever" and t.data["candidates"] == []

    class Boom:
        def search(self, *a, **k):
            raise RuntimeError("down")

    t2 = RetrievalTrace()
    _assembler(Boom()).build(date(2026, 9, 1), "hello", trace=t2)
    assert t2.data["status"] == "search_failed"


def _break_index_unavailable(tmp_path):
    r = _retriever(tmp_path, "s1", WORDS_A)
    r.vec_available = False
    return r


def _break_no_slot(tmp_path):
    r = Retriever(tmp_path / "s2.db", StubLLM())  # never indexed: no vectors for the active model
    if not r.vec_available:
        pytest.skip("sqlite-vec not available in this environment")
    return r


def _break_embed_failed(tmp_path):
    r = _retriever(tmp_path, "s3", WORDS_A)

    def down(*a, **k):
        raise RuntimeError("embed server down")

    r.embedding.embed = down
    return r


def _break_embed_dim(tmp_path):
    r = _retriever(tmp_path, "s4", WORDS_A)
    r.embedding.embed = lambda target, texts, kind="document", titles=None: [[0.1, 0.2, 0.3] for _ in texts]
    return r


def _break_sql(tmp_path):
    r = _retriever(tmp_path, "s5", WORDS_A)
    r._conn.execute("DROP TABLE dirty_documents")  # the search SQL now errors
    return r


@pytest.mark.parametrize("make, status", [
    (_break_index_unavailable, "index_unavailable"),
    (_break_no_slot, "index_unavailable"),
    (_break_embed_failed, "embed_failed"),
    (_break_embed_dim, "embed_dim_mismatch"),
    (_break_sql, "search_error"),
])
def test_trace_status_says_why_search_came_back_empty(tmp_path, make, status):
    r = make(tmp_path)
    asm = _assembler(r)
    off = asm.build(date(2026, 9, 1), "tell me about the entries")
    trace = RetrievalTrace()
    on = asm.build(date(2026, 9, 1), "tell me about the entries", trace=trace)
    assert on == off  # behaviour is the same with tracing on; only the trace knows the reason
    assert trace.data["status"] == status
    assert trace.data["candidates"] == [] and trace.data["loaded"] == []


def test_a_search_with_no_matches_is_still_ok(tmp_path):
    trace = RetrievalTrace()
    _assembler(_retriever(tmp_path, "a", WORDS_A), min_score=1.5).build(date(2026, 9, 1), "entries", trace=trace)
    assert trace.data["status"] == "ok" and trace.data["counts"]["candidates"] == 0


def test_direct_fallback_used_is_true_whenever_it_ran_with_a_separate_found_flag(tmp_path):
    r = _retriever(tmp_path, "a", WORDS_A)
    # no semantic match: the fallback runs; the store has text for the recent days -> found
    found = RetrievalTrace()
    ContextAssembler(FakeStore(today_text="Synthetic day text"), r, _cfg(min_score=1.5))         .build(date(2026, 9, 1), "entries", trace=found)
    assert found.data["direct_fallback"]["used"] is True and found.data["direct_fallback"]["found"] is True
    assert found.data["direct_fallback"]["tokens"] > 0
    # the fallback ran but the store holds nothing: used, not found, no tokens
    empty = RetrievalTrace()
    _assembler(r, min_score=1.5).build(date(2026, 9, 1), "entries", trace=empty)
    assert empty.data["direct_fallback"] == {"used": True, "found": False, "tokens": 0}
    # a semantic match means the fallback did not run
    hit = RetrievalTrace()
    _assembler(r).build(date(2026, 9, 1), "entries", trace=hit)
    assert hit.data["direct_fallback"]["used"] is False


@pytest.mark.parametrize("method, min_score", [
    ("set_params", 0.30), ("set_budget", 0.30), ("set_status", 0.30), ("add_candidate", 0.30),
    ("add_loaded", 0.30), ("finish", 0.30), ("set_direct_fallback", 1.5),
])
def test_a_failing_collector_never_fails_the_context_build(tmp_path, monkeypatch, method, min_score):
    asm = _assembler(_retriever(tmp_path, "a", WORDS_A), min_score=min_score)
    off = asm.build(date(2026, 9, 1), "tell me about the entries")

    def boom(self, *a, **k):
        raise RuntimeError("collector bug")

    monkeypatch.setattr(RetrievalTrace, method, boom)
    trace = RetrievalTrace()
    on = asm.build(date(2026, 9, 1), "tell me about the entries", trace=trace)
    assert on == off
    assert trace.failed is True
    assert "retrieval_trace" not in appmod.attach_trace({}, trace)


# ---------------- request level ----------------


@pytest.fixture
def world(tmp_path, monkeypatch):
    retrievers = {A: _retriever(tmp_path, "a", WORDS_A), B: _retriever(tmp_path, "b", WORDS_B)}
    states = {}
    for tenant, r in retrievers.items():
        asm = _assembler(r)

        def chat(messages, temperature=0.7, _t=tenant):
            return "reply-" + hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:12]

        states[tenant] = SimpleNamespace(
            assembler=asm,
            llm_main=SimpleNamespace(chat=chat, chat_stream=lambda messages, gate, temperature=0.7, alive=None: chat(messages)),
            pipeline=SimpleNamespace(log_exchange=lambda **kw: SimpleNamespace(decision="logged", xid="x1", reason="")),
        )
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.setattr(appmod, "_tenant_state", lambda request, recover=True: states[request.headers["X-Cowork-User-ID"]])
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda st: None)
    monkeypatch.setattr(appmod, "_recover_state", lambda st: None)
    monkeypatch.setattr(appmod, "_reindex_today", lambda st, day: None)
    monkeypatch.delenv(ENV_FLAG, raising=False)
    yield SimpleNamespace(client=TestClient(appmod.app), retrievers=retrievers, states=states)
    for key in [k for k in appmod.SESSIONS if k.split(":")[0] in (A, B)]:
        appmod.SESSIONS.pop(key, None)


def _post(world, tenant, headers=None, stream=False, session="s1"):
    body = {"session_id": session, "messages": [{"role": "user", "content": "tell me about the entries"}]}
    if stream:
        body.update(stream=True, diary_events=True)
    return world.client.post("/v1/chat/completions", headers={"X-Cowork-User-ID": tenant, **(headers or {})}, json=body)


def _norm(resp):
    data = resp.json()
    data.pop("created", None)
    return data


def test_off_by_default_no_field_and_identical_responses(world, monkeypatch):
    plain = _post(world, A, session="p1")
    assert plain.status_code == 200 and "retrieval_trace" not in plain.json()
    # the per-request header alone does nothing while the operator flag is unset
    with_header = _post(world, A, {HEADER: "1"}, session="p2")
    assert "retrieval_trace" not in with_header.json()
    assert _norm(with_header) == _norm(plain)
    # the operator flag alone does nothing without the header, or with any other value
    monkeypatch.setenv(ENV_FLAG, "1")
    for n, headers in enumerate(({}, {HEADER: "0"}, {HEADER: "true"}, {HEADER: ""})):
        r = _post(world, A, headers, session=f"p3-{n}")
        assert "retrieval_trace" not in r.json()
        assert _norm(r) == _norm(plain)
    # a flag that is not exactly "1" is off
    monkeypatch.setenv(ENV_FLAG, "true")
    assert "retrieval_trace" not in _post(world, A, {HEADER: "1"}, session="p4").json()


def test_on_returns_a_trace_and_the_model_input_is_unchanged(world, monkeypatch):
    plain = _post(world, A, session="q1")
    monkeypatch.setenv(ENV_FLAG, "1")
    traced = _post(world, A, {HEADER: "1"}, session="q2")
    body = traced.json()
    trace = body.pop("retrieval_trace")
    body.pop("created")
    assert body == _norm(plain)  # same reply: the model saw byte-identical messages
    assert trace["version"] == 1 and trace["status"] == "ok" and trace["rerank"] == {"present": False}
    loaded = [e["id"] for e in trace["loaded"]]
    r = world.retrievers[A]
    # the loaded ids are this tenant's chunks, and each one's text is what the model received
    assert loaded and all(any(w in _row_text(r, i) for w in WORDS_A) for i in loaded)


def test_streamed_diary_event_carries_the_trace_only_when_enabled(world, monkeypatch):
    def events(resp):
        return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]

    off = [e for e in events(_post(world, A, {HEADER: "1"}, stream=True, session="r1")) if e.get("type") == "diary"]
    assert off and "retrieval_trace" not in off[0]
    monkeypatch.setenv(ENV_FLAG, "1")
    on = [e for e in events(_post(world, A, {HEADER: "1"}, stream=True, session="r2")) if e.get("type") == "diary"]
    assert on and on[0]["retrieval_trace"]["counts"]["loaded"] == 3
    quiet = [e for e in events(_post(world, A, stream=True, session="r3")) if e.get("type") == "diary"]
    assert "retrieval_trace" not in quiet[0]


def test_a_tenant_only_ever_sees_its_own_trace(world, monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    ta = _post(world, A, {HEADER: "1"}, session="t1").json()["retrieval_trace"]
    tb = _post(world, B, {HEADER: "1"}, session="t1").json()["retrieval_trace"]
    # same chunk ids can exist in both databases; what separates them is the data behind the id
    for trace, tenant, other in ((ta, A, B), (tb, B, A)):
        mine = world.retrievers[tenant]
        for e in trace["loaded"]:
            assert any(w in _row_text(mine, e["id"]) for w in (WORDS_A if tenant == A else WORDS_B))
    assert {e["day"] for e in ta["loaded"]} == {e["day"] for e in tb["loaded"]}  # same synthetic days, own rows
    # no trace is kept anywhere that a later request (of either tenant) could receive
    assert _post(world, B, session="t2").json().get("retrieval_trace") is None
    assert _post(world, A, session="t2").json().get("retrieval_trace") is None
    for st in world.states.values():
        assert not any("trace" in name for name in vars(st.assembler))
    # there is no route that serves a trace
    assert all("trace" not in getattr(route, "path", "") for route in appmod.app.routes)


def test_two_tenants_through_the_app_each_get_only_their_own_trace_even_concurrently(world, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv(ENV_FLAG, "1")
    query = "tell me about the entries"

    def hits(tenant):
        return [(h["id"], h["score"]) for h in
                world.retrievers[tenant].search(query, top_k=16, min_score=0.30, with_ids=True)]

    expected = {A: hits(A), B: hits(B)}
    assert expected[A] != expected[B]  # the two tenants really do hold different data
    searches = {A: 0, B: 0}
    for tenant, r in world.retrievers.items():
        real = r.search

        def spy(*a, _real=real, _t=tenant, **k):
            searches[_t] += 1
            return _real(*a, **k)

        r.search = spy

    def candidates(resp):
        assert resp.status_code == 200
        return [(c["id"], c["score"]) for c in resp.json()["retrieval_trace"]["candidates"]]

    # interleaved
    for n, tenant in enumerate([A, B, B, A]):
        assert candidates(_post(world, tenant, {HEADER: "1"}, session=f"w{n}")) == expected[tenant]
    assert searches == {A: 2, B: 2}  # a tenant's request only ever touched its own index
    # concurrent
    order = [A, B] * 8
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda args: candidates(_post(world, args[1], {HEADER: "1"}, session=f"c{args[0]}")),
                            enumerate(order)))
    assert got == [expected[t] for t in order]
    assert searches == {A: 10, B: 10}
    # an untraced request after traced ones carries nothing over, for either tenant
    for tenant in (A, B):
        assert "retrieval_trace" not in _post(world, tenant, session=f"z{tenant}").json()


def test_a_failing_collector_does_not_fail_the_chat_over_http(world, monkeypatch):
    plain = _post(world, A, session="f1")
    monkeypatch.setenv(ENV_FLAG, "1")

    def boom(self, *a, **k):
        raise RuntimeError("collector bug")

    monkeypatch.setattr(RetrievalTrace, "add_candidate", boom)
    resp = _post(world, A, {HEADER: "1"}, session="f2")
    assert resp.status_code == 200 and "retrieval_trace" not in resp.json()
    assert _norm(resp) == _norm(plain)  # same reply: the model input was untouched


def test_http_trace_has_no_entry_text(world, monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    blob = json.dumps(_post(world, A, {HEADER: "1"}, session="v1").json()["retrieval_trace"])
    for needle in WORDS_A + ["Synthetic", "topic", "tell me about"]:
        assert needle not in blob
