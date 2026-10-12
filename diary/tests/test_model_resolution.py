"""#1422 / #1065: the companion always uses a model the local engine serves. Synthetic fixtures only."""
import asyncio
import json
import logging

import httpx
import pytest

from agent import llm as llm_mod
from agent.llm import EngineUnreachable, LLMClient, LLMError
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
        if self.down:
            raise httpx.ConnectError("synthetic connection refused", request=request)
        if request.method == "GET" and request.url.path == "/v1/models":
            self.listings += 1
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


def test_configured_model_that_is_not_served_falls_back_to_the_loaded_served_model(caplog):
    engine = Engine([entry("served-a", "unloaded"), entry("served-b", "loaded")])
    client = client_for(engine, "Configured-GGUF")
    with caplog.at_level(logging.INFO, logger="agent.llm"):
        assert str(client.chat([{"role": "user", "content": "Synthetic"}])) == "Synthetic answer"
        client.chat([])
    assert engine.chat_models == ["served-b", "served-b"]
    assert client.resolve_chat_model() == ("served-b", "fallback")
    chosen = [r for r in caplog.records if "chosen" in r.getMessage()]
    assert len(chosen) == 1, "the choice is logged once, not per call"
    assert chosen[0].levelno == logging.WARNING
    assert "'served-b'" in chosen[0].getMessage() and "'Configured-GGUF'" in chosen[0].getMessage()
    client.close()


def test_the_placeholder_default_is_replaced_the_same_way():
    engine = Engine([entry("served-a")])
    client = client_for(engine, "default")
    client.chat([])
    assert engine.chat_models == ["served-a"]
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


def test_streaming_chat_uses_the_resolved_model():
    engine = Engine([entry("served-a")])
    client = client_for(engine, "Configured-GGUF")
    events = []
    client.chat_stream([], events.append)
    assert engine.chat_models == ["served-a"]
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
    client.chat([])
    assert engine.chat_models == ["served-a"]
    client.close()


def test_the_served_list_is_remembered_for_60_seconds_then_read_again(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: now[0])
    engine = Engine([entry("served-a")])
    client = client_for(engine, "Configured-GGUF")
    client.chat([])
    now[0] += 59
    client.chat([])
    assert engine.listings == 1
    engine.models = [entry("served-b")]
    now[0] += 2
    client.chat([])
    assert engine.listings == 2
    assert engine.chat_models == ["served-a", "served-a", "served-b"]
    client.close()


def test_a_failed_chat_forgets_the_remembered_list():
    engine = Engine([entry("served-a")])
    client = client_for(engine, "Configured-GGUF")
    client.chat([])
    engine.models = [entry("served-b")]
    engine.models_status = 200
    # the engine swaps its models; the stale choice is rejected once, then the next call re-reads
    with pytest.raises(LLMError):
        client.chat([])
    client.chat([])
    assert engine.chat_models[-1] == "served-b"
    client.close()


def test_embedding_sections_are_never_chosen_by_launch_flags_not_names():
    engine = Engine([entry("any-name-1", "loaded", args=["llama-server", "--embedding"]),
                     entry("any-name-2", "unloaded", args=["llama-server"])])
    client = client_for(engine, "Configured-GGUF")
    client.chat([])
    assert engine.chat_models == ["any-name-2"], "none loaded among chat sections: first listed chat section"
    client.close()


def test_a_name_that_merely_looks_like_an_embedding_model_is_still_a_chat_candidate():
    engine = Engine([entry("nomic-embed-text-looking-name", "loaded", args=["llama-server"])])
    client = client_for(engine, "Configured-GGUF")
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
    main.close()
    aux.close()


def test_health_reports_the_resolved_model(monkeypatch):
    import agent.app as appmod
    from types import SimpleNamespace
    engine = Engine([entry("served-a")])
    main = client_for(engine, "Configured-GGUF")
    monkeypatch.setattr(appmod, "get_state", lambda: SimpleNamespace(llm_main=main, auth_token=""))
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.delenv("DIARY_LEGACY_USER_ID", raising=False)
    from fastapi.testclient import TestClient
    body = TestClient(appmod.app).get("/api/health").json()
    assert body["ok"] is True
    assert body["model"] == "served-a"
    assert body["model_configured"] == appmod._base_cfg.get("llm.chat_model")
    assert body["model_source"] == "fallback"
    engine.down = True
    main.invalidate_resolution()
    r = TestClient(appmod.app).get("/api/health")
    assert r.status_code == 200, "a liveness probe never fails because the engine is down"
    assert r.json()["model_source"] == "unreachable"
    main.close()


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
