"""R9 (D9): DIARY_RUST_INDEX=shadow against a stand-in for POST /internal/reindex with plan_only.
Python re-indexes as before; the Rust plan of each dirty document is compared with the plan Python
applies, only the outcome is logged and counted (never text), and nothing the shadow does changes
the re-index. Synthetic data only."""
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

import agent.app as appmod
from agent import rust_answer, rust_forward, rust_index
from agent.embedding import Target
from agent.managed_storage import ManagedCorpusBackend
from agent.retrieval import ReindexPlan
from agent.workspace_files import MemoryBackend

TENANT = "6c1d8e2f-3a4b-4c5d-8e6f-708192a3b4c5"
SECRET_WORD = "QA-synthetic-larkspur"
DOCS = ["Entries/2026/September/September 1, 2026.md", "AI Memory/qa.md"]
TARGET = Target(None, "qa-embed-a", "plain")
REAL_SHADOW = rust_index.shadow


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    monkeypatch.setenv("DIARY_RUST_INDEX", "shadow")
    monkeypatch.delenv("DIARY_RUST_ANSWER", raising=False)
    monkeypatch.setattr(rust_index, "SHADOW_STATS", {k: 0 for k in rust_index.SHADOW_OUTCOMES})


class FakeServer:
    base_url = "http://127.0.0.1:1"

    def __init__(self, ready=True):
        self._ready = ready

    def control(self):
        return "synthetic-control-token", self._ready


class FakeManaged(ManagedCorpusBackend):
    def __init__(self, tenant):
        self.tenant = tenant
        self.files = {d: f"## {SECRET_WORD} text of {d}\n" for d in DOCS}

    def get_text(self, path):
        return self.files[path], None


def py_plan(document, vector=(0.25, 0.5), pending=False):
    h = f"hash-{document[-6:]}"
    return ReindexPlan(document, [("2026-09-01", f"{SECRET_WORD} header", f"{SECRET_WORD} body", h)],
                       {h: list(vector)}, pending, TARGET)


def rust_plan(document, vector=(0.25, 0.5), pending=False, **over):
    h = f"hash-{document[-6:]}"
    return {"document": document, "rows": [["2026-09-01", f"{SECRET_WORD} header", h]],
            "embeddings": {h: list(vector)}, "pending": pending, "target": TARGET.to_json(), "error": None, **over}


def state(order, backend=None):
    def prepare(document, text):
        order.append(("prepare", document))
        return py_plan(document)

    def apply(plan):
        order.append(("apply", plan.file_name))
        return 1

    config = SimpleNamespace(get=lambda: {"active": TARGET.to_json()})
    retrieval = SimpleNamespace(embedding=SimpleNamespace(config=config), prepare_reindex=prepare, apply_reindex=apply,
                                reindex_file=lambda d, t: order.append(("reindex_file", d)), pending_embeddings=False)
    store = SimpleNamespace(backend=backend or FakeManaged(TENANT), _write_lock=__import__("threading").RLock())
    cleared = []
    journal = SimpleNamespace(dirty_documents=lambda: [d for d in DOCS if d not in cleared], clear_dirty=cleared.append)
    return SimpleNamespace(store=store, retrieval=retrieval, journal=journal, tenant_id=TENANT, cleared=cleared)


def run(monkeypatch, handler, st=None, order=None):
    order = [] if order is None else order
    st = st or state(order)
    seen = []

    def recording(request):
        seen.append({"body": json.loads(request.content), "headers": dict(request.headers),
                     "tenant": request.url.params.get("tenant")})
        order.append(("rust",))
        return handler(request)
    monkeypatch.setattr(rust_forward, "_server", FakeServer())
    monkeypatch.setattr(appmod.rust_index, "shadow",
                        lambda *a, **k: REAL_SHADOW(*a, transport=httpx.MockTransport(recording), **k))
    import agent.workspace_import as wi
    monkeypatch.setattr(wi, "drain_index_outbox", lambda *a: None)
    appmod._reindex_dirty(st)
    return order, seen, st


def reply(*docs, status=200, headers=None):
    return lambda r: httpx.Response(status, headers=headers or {}, json={"documents": list(docs)})


def test_matching_plans_are_counted_and_python_still_indexes(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    order, seen, st = run(monkeypatch, reply(*(rust_plan(d) for d in DOCS)))
    # One plan call first, then Python's own prepare + apply per document; marks cleared as before.
    assert order == [("rust",)] + [step for d in DOCS for step in (("prepare", d), ("apply", d))]
    assert st.cleared == DOCS
    (req,) = seen
    assert req["tenant"] == TENANT and req["body"]["plan_only"] is True and req["body"]["documents"] == DOCS
    assert req["body"]["embedding"] == {"active": TARGET.to_json()}
    assert "x-noevia-diary-writer" not in req["headers"]
    assert rust_index.shadow_stats()["match"] == 2
    assert SECRET_WORD not in caplog.text


@pytest.mark.parametrize("rust,why", [
    (lambda d: rust_plan(d, vector=(0.25, 0.5000001)), "vectors differ"),
    (lambda d: rust_plan(d, vector=(0.25,)), "widths differ"),
    (lambda d: rust_plan(d, pending=True), "pending differs"),
    (lambda d: rust_plan(d, rows=[]), "row counts differ"),
    (lambda d: rust_plan(d, rows=[["2026-09-02", f"{SECRET_WORD} header", "hash-" + d[-6:]]]), "row 0: day differs"),
    (lambda d: rust_plan(d, target={"base_url": None, "model": "qa-embed-b", "profile": "plain"}), "targets differ"),
    (lambda d: rust_plan(d, embeddings={}), "embedded chunks differ"),
    (lambda d: rust_plan(d, error="read"), "could not plan"),
])
def test_a_difference_is_a_mismatch_named_without_content(monkeypatch, caplog, rust, why):
    caplog.set_level(logging.DEBUG)
    order, _, st = run(monkeypatch, reply(*(rust(d) for d in DOCS)))
    assert st.cleared == DOCS and ("apply", DOCS[1]) in order
    assert rust_index.shadow_stats()["mismatch"] == 2
    assert why in caplog.text and SECRET_WORD not in caplog.text


def test_float_parsing_noise_is_not_a_mismatch():
    assert rust_index.compare_plan(py_plan(DOCS[0], vector=(0.1 + 0.2, 1e-3)),
                                   rust_plan(DOCS[0], vector=(0.3, 1e-3 + 1e-16))) is None


@pytest.mark.parametrize("handler,outcome", [
    (reply(status=421, headers={rust_forward.DECLINE_HEADER.decode(): "1"}), "declined"),
    (reply(status=500), "refused"),
    (lambda r: httpx.Response(200, content=b"not json"), "failed"),
    (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("synthetic")), "failed"),
])
def test_no_complete_reply_changes_nothing(monkeypatch, handler, outcome):
    order, _, st = run(monkeypatch, handler)
    assert order == [("rust",)] + [("reindex_file", d) for d in DOCS]
    assert st.cleared == DOCS
    assert rust_index.shadow_stats()[outcome] == 1


def test_off_on_and_inapplicable_cases_never_call_it(monkeypatch):
    for mode, expected in (("", "reindex_file"), ("bogus", "reindex_file")):
        monkeypatch.setenv("DIARY_RUST_INDEX", mode)
        order, seen, _ = run(monkeypatch, reply())
        assert seen == [] and [o[0] for o in order] == [expected, expected]
    monkeypatch.setenv("DIARY_RUST_INDEX", "shadow")
    order, seen, _ = run(monkeypatch, reply(), st=state([], backend=MemoryBackend({d: "x" for d in DOCS})))
    assert seen == [] and rust_index.shadow_stats()["skipped"] == 1
    st = state([])
    st.tenant_id = "legacy"
    _, seen, _ = run(monkeypatch, reply(), st=st)
    assert seen == []
    monkeypatch.setattr(rust_forward, "_server", None)
    assert REAL_SHADOW(state([]), TENANT, DOCS) is None


def test_shadow_sends_the_start_up_config_and_takes_no_writer_role():
    assert rust_index.shadowing() and rust_index.mode() == "shadow"
    assert rust_answer.answer_config(__import__("agent.config", fromlist=["Config"]).Config({}), 0) is not None
    # The `on` hand-over is not taken in shadow mode.
    assert rust_index.reindex_dirty(state([]), TENANT, server_lookup=lambda: FakeServer()) is False


def test_a_large_dirty_set_is_not_compared(monkeypatch):
    many = [f"Entries/2026/September/September {d}, 2026.md" for d in range(1, 10)]
    called = []
    st = state([])
    st.journal.dirty_documents = lambda: list(many)
    st.store.backend.files.update({d: "x" for d in many})
    monkeypatch.setattr(rust_forward, "_server", FakeServer())
    assert REAL_SHADOW(st, TENANT, many, transport=httpx.MockTransport(lambda r: called.append(1))) is None
    assert called == [] and rust_index.shadow_stats()["skipped"] == 1
    assert REAL_SHADOW(st, TENANT, many[:rust_index.SHADOW_MAX_DOCUMENTS],
                       transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"documents": []}))) is not None
