"""#1422 / #1065: the companion always uses a model the local engine serves. Synthetic fixtures only."""
import asyncio
import json
import logging

import httpx
import pytest

from agent import llm as llm_mod
from agent.llm import EngineUnreachable, LLMClient, LLMError, ModelNotServed
from agent.streaming import exchange_stream

BASE = "http://synthetic.invalid/v1"


def entry(model_id, state="loaded", args=None):
    status = {"value": state}
    if args is not None:
        status["args"] = args
    return {"id": model_id, "object": "model", "status": status}


class Engine:
    """A synthetic router: /v1/models plus chat completions, recording what was asked."""

    def __init__(self, models, models_status=200):
        self.models = models
        self.models_status = models_status
        self.listings = 0
        self.chat_models = []
        self.down = False

    def __call__(self, request):
        if request.method == "GET" and request.url.path == "/v1/models":
            self.listings += 1  # a refused attempt is still a /models call
        if self.down:
            raise httpx.ConnectError("synthetic connection refused", request=request)
        if request.method == "GET" and request.url.path == "/v1/models":
            if self.models_status != 200:
                return httpx.Response(self.models_status, text="no")
            return httpx.Response(200, json={"object": "list", "data": self.models})
        body = json.loads(request.content)
        self.chat_models.append(body["model"])
        if body["model"] not in {m["id"] for m in self.models}:
            return httpx.Response(400, json={"error": {"message": "model not found"}})
        if body.get("stream"):
            text = "data: " + json.dumps({"choices": [{"delta": {"content": "Synthetic answer"}, "finish_reason": "stop"}]}) + "\n\ndata: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Synthetic answer"}}]})


def client_for(engine, configured, **kwargs):
    client = LLMClient(BASE, chat_model=configured, max_retries=1, resolve_served_model=True, **kwargs)
    client._client.close()
    client._client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(engine))
    return client


def test_configured_model_that_is_not_served_is_refused_and_nothing_is_sent(caplog):
    """#2099: never "first loaded" or "first listed": the router would load a model nobody chose."""
    engine = Engine([entry("served-a", "unloaded"), entry("served-b", "loaded")])
    client = client_for(engine, "Configured-GGUF")
    with caplog.at_level(logging.INFO, logger="agent.llm"):
        with pytest.raises(ModelNotServed) as refused:
            client.chat([{"role": "user", "content": "Synthetic"}])
        with pytest.raises(ModelNotServed):
            client.chat_stream([], lambda e: None)
    assert engine.chat_models == [], "no chat request without a served, explicit id"
    assert refused.value.setting == "LLM_CHAT_MODEL"
    assert "LLM_CHAT_MODEL" in str(refused.value) and "Configured-GGUF" not in str(refused.value)
    assert "Configured-GGUF" not in caplog.text, "the setting key is named, never its value"
    assert len([r for r in caplog.records if "LLM_CHAT_MODEL" in r.getMessage()]) == 1, "logged once, not per call"
    client.close()


def test_a_refusal_is_not_remembered_so_a_fixed_setting_recovers_at_once():
    engine = Engine([entry("served-a")])
    client = client_for(engine, "Configured-GGUF")
    with pytest.raises(ModelNotServed):
        client.chat([])
    engine.models = [entry("served-a"), entry("Configured-GGUF", "unloaded")]
    client.chat([])
    assert engine.chat_models == ["Configured-GGUF"]
    client.close()


def test_an_explicit_fallback_id_is_used_when_the_configured_one_is_not_served(caplog):
    engine = Engine([entry("served-a", "loaded"), entry("served-b", "unloaded")])
    client = client_for(engine, "Configured-GGUF", fallback_model="served-b")
    with caplog.at_level(logging.INFO, logger="agent.llm"):
        client.chat([])
        client.chat([])
    assert engine.chat_models == ["served-b", "served-b"], "the named fallback, not the loaded model"
    assert client.resolve_chat_model() == ("served-b", "fallback")
    warned = [r for r in caplog.records if r.levelno == logging.WARNING and "LLM_CHAT_MODEL" in r.getMessage()]
    assert len(warned) == 1 and "Configured-GGUF" not in caplog.text
    client.close()


def test_a_fallback_id_that_is_not_served_either_is_refused_naming_both_keys():
    engine = Engine([entry("served-a", "loaded")])
    client = client_for(engine, "Configured-GGUF", fallback_model="Also-missing")
    with pytest.raises(ModelNotServed, match="LLM_CHAT_FALLBACK_MODEL") as refused:
        client.chat([])
    assert "Also-missing" not in str(refused.value) and "Configured-GGUF" not in str(refused.value)
    assert engine.chat_models == []
    client.close()


def test_the_placeholder_default_is_refused_when_it_is_not_served():
    engine = Engine([entry("served-a")])
    client = client_for(engine, "default")
    with pytest.raises(ModelNotServed):
        client.chat([])
    assert engine.chat_models == []
    client.close()


def test_an_empty_model_setting_is_refused_with_or_without_model_listing():
    for resolve in (True, False):
        engine = Engine([entry("served-a")])
        client = LLMClient(BASE, chat_model="  ", max_retries=1, resolve_served_model=resolve,
                           setting_key="LLM_AUX_MODEL")
        client._client.close()
        client._client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(engine))
        with pytest.raises(ModelNotServed, match="LLM_AUX_MODEL"):
            client.chat([])
        assert engine.chat_models == []
        client.close()


def test_a_configured_id_that_is_only_an_embedding_section_is_not_a_chat_model():
    engine = Engine([entry("embed-only", "loaded", args=["--embedding"]), entry("served-a")])
    client = client_for(engine, "embed-only")
    with pytest.raises(ModelNotServed):
        client.chat([])
    assert engine.chat_models == []
    client.close()


def test_configured_model_that_is_served_is_used_even_when_another_is_loaded():
    engine = Engine([entry("served-a", "loaded"), entry("configured-id", "unloaded")])
    client = client_for(engine, "configured-id")
    client.chat([])
    events = []
    client.chat_stream([], events.append)
    assert engine.chat_models == ["configured-id", "configured-id"]
    assert client.resolve_chat_model() == ("configured-id", "configured")
    client.close()


def test_streaming_chat_uses_the_configured_served_model():
    engine = Engine([entry("served-a"), entry("Configured-GGUF", "unloaded")])
    client = client_for(engine, "Configured-GGUF")
    events = []
    client.chat_stream([], events.append)
    assert engine.chat_models == ["Configured-GGUF"]
    client.close()


def test_router_down_is_a_clear_logged_error_and_no_chat_is_attempted(caplog):
    engine = Engine([entry("served-a")])
    engine.down = True
    client = client_for(engine, "Configured-GGUF")
    with caplog.at_level(logging.ERROR, logger="agent.llm"):
        with pytest.raises(EngineUnreachable, match="unreachable"):
            client.chat([{"role": "user", "content": "Synthetic secret diary text"}])
    assert engine.chat_models == []
    assert any("could not be reached" in r.getMessage() for r in caplog.records)
    assert "Synthetic secret diary text" not in caplog.text
    # recovery: the failure is not remembered
    engine.down = False
    engine.models = [entry("Configured-GGUF")]
    client.chat([])
    assert engine.chat_models == ["Configured-GGUF"]
    client.close()


def test_the_served_list_is_remembered_for_60_seconds_then_read_again(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: now[0])
    engine = Engine([entry("served-a")])
    client = client_for(engine, "served-a")
    client.chat([])
    now[0] += 59
    client.chat([])
    assert engine.listings == 1
    engine.models = [entry("served-b")]
    now[0] += 2
    with pytest.raises(ModelNotServed):
        client.chat([])
    assert engine.listings == 2
    assert engine.chat_models == ["served-a", "served-a"]
    client.close()


def test_a_failed_chat_forgets_the_remembered_list():
    engine = Engine([entry("served-a")])
    client = client_for(engine, "served-a")
    client.chat([])
    engine.models = [entry("served-b")]
    # the engine swaps its models; the stale choice is rejected once, then the next call re-reads
    with pytest.raises(LLMError):
        client.chat([])
    with pytest.raises(ModelNotServed):
        client.chat([])
    assert engine.chat_models == ["served-a", "served-a"], "one stale send, then nothing once the list is re-read"
    client.close()


def test_a_name_that_merely_looks_like_an_embedding_model_is_still_a_chat_candidate():
    engine = Engine([entry("nomic-embed-text-looking-name", "loaded", args=["llama-server"])])
    client = client_for(engine, "nomic-embed-text-looking-name")
    client.chat([])
    assert engine.chat_models == ["nomic-embed-text-looking-name"]
    client.close()


def test_an_engine_serving_no_chat_model_is_a_clear_error():
    engine = Engine([entry("only-embed", "loaded", args=["--embedding"])])
    client = client_for(engine, "Configured-GGUF")
    with pytest.raises(LLMError, match="no chat model"):
        client.chat([])
    assert engine.chat_models == []
    client.close()


def test_an_endpoint_that_cannot_list_models_keeps_the_configured_id():
    engine = Engine([entry("served-a")], models_status=404)
    client = client_for(engine, "served-a")
    client.chat([])
    assert engine.chat_models == ["served-a"]
    assert client.resolve_chat_model() == ("served-a", "unverified")
    client.close()


def test_a_client_that_did_not_opt_in_never_lists_models():
    engine = Engine([entry("served-a")])
    client = LLMClient(BASE, chat_model="Configured-GGUF", max_retries=1)
    client._client.close()
    client._client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(engine))
    with pytest.raises(LLMError):
        client.chat([])
    assert engine.listings == 0 and engine.chat_models == ["Configured-GGUF"]
    client.close()


def test_the_aux_client_follows_the_chat_models_choice_when_its_own_model_is_not_served():
    engine = Engine([entry("served-a", "unloaded"), entry("served-b", "loaded"), entry("served-c", "unloaded")])
    main = client_for(engine, "served-c")
    aux = client_for(engine, "Configured-aux", prefer_model_from=main)
    aux.chat([])
    assert engine.chat_models == ["served-c"], "chat configured id is served, so aux follows it, not the loaded one"
    aux.fallback_model = "served-a"
    aux.invalidate_resolution()
    aux.chat([])
    assert engine.chat_models[-1] == "served-a", "its own explicit fallback comes before the chat model's id"
    main.close()
    aux.close()


def test_health_reports_the_resolved_model(monkeypatch):
    import agent.app as appmod
    from types import SimpleNamespace
    engine = Engine([entry("served-a")])
    main = client_for(engine, "Configured-GGUF", fallback_model="served-a")
    monkeypatch.setattr(appmod, "get_state", lambda: SimpleNamespace(llm_main=main, auth_token=""))
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.delenv("DIARY_LEGACY_USER_ID", raising=False)
    from fastapi.testclient import TestClient
    body = TestClient(appmod.app).get("/api/health").json()
    assert body["ok"] is True
    assert body["model"] == "served-a"
    assert body["model_configured"] == appmod._base_cfg.get("llm.chat_model")
    assert body["model_source"] == "fallback"
    assert body["model_warnings"] == ["LLM_CHAT_MODEL"], "a key, never the value"
    engine.down = True
    main.invalidate_resolution()
    r = TestClient(appmod.app).get("/api/health")
    assert r.status_code == 200, "a liveness probe never fails because the engine is down"
    assert r.json()["model_source"] == "unreachable"
    main.close()


def test_health_and_startup_name_the_setting_key_of_an_unserved_id(monkeypatch, caplog):
    import agent.app as appmod
    from types import SimpleNamespace
    engine = Engine([entry("served-a")])
    main = client_for(engine, "Configured-GGUF")
    aux = client_for(engine, "Configured-aux", setting_key="LLM_AUX_MODEL")
    state = SimpleNamespace(llm_main=main, llm_aux=aux, auth_token="")
    monkeypatch.setattr(appmod, "get_state", lambda: state)
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.delenv("DIARY_LEGACY_USER_ID", raising=False)
    from fastapi.testclient import TestClient
    body = TestClient(appmod.app).get("/api/health").json()
    assert body["ok"] is True and body["model_source"] == "not_served"
    assert body["model_warnings"] == ["LLM_CHAT_MODEL", "LLM_AUX_MODEL"]
    with caplog.at_level(logging.WARNING):
        appmod._model_startup_check()
    startup = [r.getMessage() for r in caplog.records if "does not serve" in r.getMessage()]
    assert any("LLM_CHAT_MODEL" in m for m in startup) and any("LLM_AUX_MODEL" in m for m in startup)
    assert "Configured-GGUF" not in caplog.text and "Configured-aux" not in caplog.text
    assert engine.chat_models == []
    # a served id is warning-free
    engine.models = [entry("Configured-GGUF"), entry("Configured-aux")]
    main.invalidate_resolution()
    aux.invalidate_resolution()
    assert TestClient(appmod.app).get("/api/health").json()["model_warnings"] == []
    main.close()
    aux.close()


def test_streamed_failure_is_logged_with_types_and_status_only(caplog):
    secret = "Synthetic private diary sentence"

    def work(emit):
        request = httpx.Request("POST", "http://router.invalid/v1/chat/completions")
        response = httpx.Response(400, request=request, text=secret)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"chat failed after 1 attempts: {exc} {secret}") from exc

    async def run():
        stream = exchange_stream(work).body_iterator
        return [chunk async for chunk in stream]

    with caplog.at_level(logging.ERROR, logger="diary.streaming"):
        chunks = asyncio.run(run())
    assert any('"type": "error"' in c for c in chunks)
    messages = [r.getMessage() for r in caplog.records]
    assert any("LLMError" in m and "HTTPStatusError status=400" in m for m in messages)
    assert secret not in caplog.text and "router.invalid" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


# ---------------- the status contract noevia core's "nothing was saved" claim relies on ----------------


def test_status_contract_pre_save_texts_and_saving_began_before_any_store_write():
    """noevia-core server/diary-stream.cjs (PRE_SAVE_STATUSES, MODEL_STATUS) claims "nothing was saved"
    when the companion fails after one of these statuses and before any other status. Any status text
    NOT in that set counts as saving having begun, so the companion must (a) send those texts exactly,
    and (b)/(c) send "Checking whether to save this exchange…" before it touches the store at all."""
    from types import SimpleNamespace
    import yaml
    import agent.app as appmod
    from pathlib import Path
    from agent.pipeline import LoggingPipeline

    # Copied verbatim from ~/noevia-work/w1422-core/server/diary-stream.cjs lines 13-16 (noevia-core):
    # MODEL_STATUS and PRE_SAVE_STATUSES. Change both sides together.
    core_model_status = "Generating companion response…"
    core_pre_save = {"Opening diary context…", "Reading diary entries and memory…", core_model_status}
    saving_began = "Checking whether to save this exchange…"  # pipeline.py log_exchange; NOT in core's set

    timeline = []  # ("status", text) | ("store", method) | ("model", kind), in order

    class RecordingStore:
        def get_standing_sections_text(self, max_chars=0):
            timeline.append(("store", "get_standing_sections_text"))
            return ""

        def update_standing_sections(self, *a, **k):
            timeline.append(("store", "update_standing_sections"))
            return None

        def log_exchange(self, **k):
            timeline.append(("store", "log_exchange"))
            return "xid-synthetic"

    class Main:
        def chat_stream(self, messages, emit, temperature=0.7, alive=None):  # alive: #1505 deadline progress hook
            timeline.append(("model", "chat_stream"))
            return llm_mod.ChatReply("A synthetic reply.\n\n[LOG: ok]")

    class Aux:
        def chat(self, messages, **k):
            prompt = messages[-1]["content"]
            timeline.append(("model", "aux"))
            return "LOG" if "Verdict (LOG or SKIP)" in prompt else ("NO" if "UPDATE or NO" in prompt else "Synthetic prose.")

    templates = yaml.safe_load((Path(appmod.__file__).resolve().parent.parent / "config" / "prompts" / "logging.md").read_text(encoding="utf-8"))
    store = RecordingStore()
    st = SimpleNamespace(
        assembler=SimpleNamespace(build=lambda day, message, session_turns=None: [{"role": "user", "content": message}]),
        llm_main=Main(),
        pipeline=LoggingPipeline(store, Main(), Aux(), templates),
        store=store,
    )

    def work(emit):
        def record(event):
            if event.get("type") == "status":
                timeline.append(("status", event["text"]))
            emit(event)
        return appmod._run_exchange(st, "Synthetic message", "contract-test", "legacy", background=False, emit=record)

    async def run():
        return [c async for c in exchange_stream(work).body_iterator]

    import unittest.mock as mock
    with mock.patch.object(appmod, "_reindex_dirty", lambda s: timeline.append(("store", "reindex_dirty"))):
        chunks = asyncio.run(run())
    appmod.SESSIONS.pop("legacy:contract-test", None)

    # What the browser/core actually receive over the wire.
    wire = [json.loads(c[6:]) for c in chunks if c.startswith("data: ")]
    wire_statuses = [e["text"] for e in wire if e["type"] == "status"]
    assert wire[-1]["type"] == "done" and "diary" in [e["type"] for e in wire]

    # (a) the pre-save texts are emitted exactly as core spells them, in order, before anything else.
    assert wire_statuses[:3] == ["Opening diary context…", "Reading diary entries and memory…", core_model_status]
    assert set(wire_statuses[:3]) == core_pre_save

    # (b) the saving-began status follows the model status and is not one of core's pre-save texts.
    assert saving_began not in core_pre_save
    assert wire_statuses[3] == saving_began
    assert all(t not in core_pre_save for t in wire_statuses[3:]), "every later status counts as saving to core"

    # (c) no store access (reads included) happens between the pre-save statuses and "Checking whether
    # to save", and no write of any kind precedes it. The real write is log_exchange, after it.
    order = timeline
    began = order.index(("status", saving_began))
    before = order[:began]
    assert [e for e in before if e[0] == "store" and e[1] in ("log_exchange", "update_standing_sections")] == []
    assert ("store", "log_exchange") in order[began:]
    assert order.index(("status", core_model_status)) < order.index(("model", "chat_stream"))
    assert order.index(("model", "chat_stream")) < began


def _health_app(monkeypatch, state):
    import agent.app as appmod
    monkeypatch.setattr(appmod, "get_state", lambda: state)
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.delenv("DIARY_LEGACY_USER_ID", raising=False)
    return appmod


def test_one_health_check_lists_the_engine_once_with_chat_and_aux_on_it(monkeypatch):
    """#112: /api/health resolved the chat model twice (warnings, then the model fields), and the aux
    client listed the same engine again. A check now lists the engine once: on success, when the id
    is not served and when the engine is down."""
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    engine = Engine([entry("Configured-GGUF"), entry("Configured-aux")])
    main = client_for(engine, "Configured-GGUF")
    aux = client_for(engine, "Configured-aux", setting_key="LLM_AUX_MODEL", prefer_model_from=main)
    appmod = _health_app(monkeypatch, SimpleNamespace(llm_main=main, llm_aux=aux, auth_token=""))

    def check():
        main.invalidate_resolution()
        aux.invalidate_resolution()
        before = engine.listings
        body = TestClient(appmod.app).get("/api/health").json()
        return body, engine.listings - before

    body, listed = check()
    assert body["model"] == "Configured-GGUF" and body["model_source"] == "configured"
    assert body["model_warnings"] == [] and listed == 1
    engine.models = [entry("served-a")]
    body, listed = check()
    assert body["model_source"] == "not_served"
    assert body["model_warnings"] == ["LLM_CHAT_MODEL", "LLM_AUX_MODEL"]
    assert listed == 1
    engine.down = True
    body, listed = check()
    assert body["model_source"] == "unreachable" and listed == 1
    main.close()
    aux.close()


def test_one_health_check_lists_a_lone_chat_client_once_when_refused_or_down(monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    engine = Engine([entry("served-a")])
    main = client_for(engine, "Configured-GGUF")
    appmod = _health_app(monkeypatch, SimpleNamespace(llm_main=main, auth_token=""))
    before = engine.listings
    body = TestClient(appmod.app).get("/api/health").json()
    assert body["model_source"] == "not_served" and engine.listings - before == 1
    engine.down = True
    before = engine.listings
    body = TestClient(appmod.app).get("/api/health").json()
    assert body["model_source"] == "unreachable" and engine.listings - before == 1
    main.close()


def test_the_listing_is_shared_only_inside_one_listing_block():
    engine = Engine([entry("served-a")])
    main = client_for(engine, "served-a")
    other = client_for(engine, "served-a")
    with llm_mod.one_listing_per_check():
        main.chat([])
        other.chat([])
    assert engine.listings == 1
    main.invalidate_resolution()
    other.invalidate_resolution()
    main.chat([])
    other.chat([])
    assert engine.listings == 3, "outside the block each client lists for itself"
    main.close()
    other.close()
