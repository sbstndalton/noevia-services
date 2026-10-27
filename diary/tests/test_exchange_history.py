"""#1505: only completed, answered exchanges are replayed to the model; the new message is always the
last user turn. Synthetic fixtures only."""
import json
from datetime import date
from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.context import ContextAssembler


class FakeStore:
    def __init__(self, today_text=""):
        self._today = today_text

    def get_day_text(self, day, max_chars=None):
        return self._today

    def get_standing_sections_text(self, max_chars=None):
        return ""


def _cfg():
    return Config({
        "context": {"max_today_tokens": 4500, "max_standing_tokens": 2200},
        "retrieval": {"top_k": 8, "min_score": 0.30, "max_context_tokens": 5200},
        "prompts": {"system_path": ""},
    })


def test_unanswered_and_empty_exchanges_are_dropped_from_history():
    from agent.context import completed_exchanges

    turns = [
        {"role": "user", "content": "Synthetic question one"},
        {"role": "assistant", "content": ""},  # stuck: the reply never arrived
        {"role": "user", "content": "Synthetic question two"},
        {"role": "companion", "content": "Synthetic answer two"},  # stored role name
        {"role": "user", "content": "Synthetic question three"},  # unanswered, trailing
    ]
    assert completed_exchanges(turns) == [
        {"role": "user", "content": "Synthetic question two"},
        {"role": "companion", "content": "Synthetic answer two"},
    ]


def test_orphan_replies_and_non_chat_roles_are_dropped():
    from agent.context import completed_exchanges

    turns = [
        {"role": "assistant", "content": "Orphan reply with no question"},
        {"role": "tool", "name": "synthetic_tool", "content": "synthetic result"},
        {"role": "user", "content": "Synthetic question"},
        {"role": "companion", "content": "Synthetic answer"},
    ]
    assert completed_exchanges(turns) == [
        {"role": "user", "content": "Synthetic question"},
        {"role": "companion", "content": "Synthetic answer"},
    ]


def test_assembler_replays_only_completed_turns_and_keeps_the_new_message_last():
    asm = ContextAssembler(FakeStore(today_text="TODAY"), None, _cfg())
    asm._system_prompt = lambda: "SYSTEM"
    turns = [
        {"role": "user", "content": "STALE QUESTION"},
        {"role": "companion", "content": ""},
        {"role": "user", "content": "Answered question"},
        {"role": "companion", "content": "Answered reply"},
    ]
    messages = asm.build(date(2026, 9, 1), "NEW MESSAGE", session_turns=turns)
    assert messages[-1] == {"role": "user", "content": "NEW MESSAGE"}
    convo = messages[1]["content"]
    assert "STALE QUESTION" not in convo
    assert "Answered question" in convo and "Answered reply" in convo


@pytest.mark.parametrize("stream", [True, False])
def test_v1_seeds_a_new_session_only_with_completed_exchanges(monkeypatch, stream):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    tenant = "11111111-1111-4111-8111-111111111111"
    seen = {}
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.setattr(appmod, "_tenant_state", lambda request, recover=True: SimpleNamespace(store=SimpleNamespace(require_write_transform=lambda: None)))

    def exchange(st, message, sid, owner, now, day, background, extra, emit=None):
        seen["message"] = message
        seen["turns"] = [dict(t) for t in appmod.SESSIONS[f"{owner}:{sid}"]["turns"]]
        return {"reply": "Synthetic answer", "decision": "skipped", "xid": None, "reasoning": ""}

    monkeypatch.setattr(appmod, "_run_exchange", exchange)
    body = {"session_id": "history-seed", "stream": stream, "diary_events": stream, "messages": [
        {"role": "user", "content": "STALE QUESTION"},  # never answered
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Answered question"},
        {"role": "assistant", "content": "Answered reply"},
        {"role": "user", "content": "NEW MESSAGE"},
    ]}
    try:
        r = TestClient(appmod.app).post("/v1/chat/completions", headers={"X-Cowork-User-ID": tenant}, json=body)
        assert r.status_code == 200, r.text
        assert seen["message"] == "NEW MESSAGE"
        assert seen["turns"] == [{"role": "user", "content": "Answered question"},
                                 {"role": "assistant", "content": "Answered reply"}]
    finally:
        appmod.SESSIONS.pop(f"{tenant}:history-seed", None)


def test_v1_new_message_is_the_last_user_turn_even_when_the_thread_ends_with_an_assistant_turn(monkeypatch):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    tenant = "11111111-1111-4111-8111-111111111111"
    seen = {}
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)
    monkeypatch.setattr(appmod, "_tenant_state", lambda request, recover=True: SimpleNamespace(store=SimpleNamespace(require_write_transform=lambda: None)))

    def exchange(st, message, sid, owner, now, day, background, extra, emit=None):
        seen["message"] = message
        seen["turns"] = [dict(t) for t in appmod.SESSIONS[f"{owner}:{sid}"]["turns"]]
        return {"reply": "Synthetic answer", "decision": "skipped", "xid": None, "reasoning": ""}

    monkeypatch.setattr(appmod, "_run_exchange", exchange)
    body = {"session_id": "history-tail", "messages": [
        {"role": "user", "content": "Answered question"},
        {"role": "assistant", "content": "Answered reply"},
        {"role": "user", "content": "NEW MESSAGE"},
        {"role": "assistant", "content": ""},  # client-side placeholder after the new message
    ]}
    try:
        r = TestClient(appmod.app).post("/v1/chat/completions", headers={"X-Cowork-User-ID": tenant}, json=body)
        assert r.status_code == 200, r.text
        assert seen["message"] == "NEW MESSAGE"
        assert seen["turns"] == [{"role": "user", "content": "Answered question"},
                                 {"role": "assistant", "content": "Answered reply"}]
    finally:
        appmod.SESSIONS.pop(f"{tenant}:history-tail", None)


def _exchange_state(store):
    from agent.llm import ChatReply
    from agent.pipeline import LoggingPipeline

    class Main:
        def chat(self, messages, **kwargs):
            return ChatReply("Synthetic reply.\n[LOG: ok]")

    class Aux:
        def chat(self, messages, **kwargs):
            return "LOG" if messages[-1]["content"].startswith("CLASSIFY") else "Synthetic summary."

    templates = {"skip_classifier": "CLASSIFY {user_message} {assistant_message}", "summarizer": "SUMMARIZE {assistant_message}"}
    return SimpleNamespace(
        assembler=SimpleNamespace(build=lambda day, message, session_turns=None: [{"role": "user", "content": message}]),
        llm_main=Main(),
        pipeline=LoggingPipeline(store, Main(), Aux(), templates),
        store=store,
    )


def test_an_exchange_whose_save_failed_is_not_remembered_as_history(monkeypatch):
    import agent.app as appmod

    class FailingStore:
        def require_write_transform(self):
            pass  # This test double represents an available Python transform.

        def get_standing_sections_text(self, max_chars=0):
            return ""

        def log_exchange(self, **kwargs):
            raise OSError("synthetic write failure")

    monkeypatch.setattr(appmod, "_reindex_dirty", lambda st: None)
    try:
        result = appmod._run_exchange(_exchange_state(FailingStore()), "Synthetic question", "history-save-fails", "legacy", background=False)
        assert result["decision"] == "error"
        assert appmod.SESSIONS["legacy:history-save-fails"]["turns"] == []
    finally:
        appmod.SESSIONS.pop("legacy:history-save-fails", None)


def test_a_saved_exchange_is_remembered_as_a_completed_pair(monkeypatch):
    import agent.app as appmod

    class SavingStore:
        def require_write_transform(self):
            pass  # This test double represents an available Python transform.

        def get_standing_sections_text(self, max_chars=0):
            return ""

        def update_standing_sections(self, *a, **k):
            return None

        def log_exchange(self, **kwargs):
            return "xid-synthetic"

    monkeypatch.setattr(appmod, "_reindex_dirty", lambda st: None)
    try:
        result = appmod._run_exchange(_exchange_state(SavingStore()), "Synthetic question", "history-saved", "legacy", background=False)
        assert result["decision"] == "logged"
        assert appmod.SESSIONS["legacy:history-saved"]["turns"] == [
            {"role": "user", "content": "Synthetic question"},
            {"role": "companion", "content": "Synthetic reply."},
        ]
    finally:
        appmod.SESSIONS.pop("legacy:history-saved", None)


def test_local_exchange_seeds_only_completed_history(monkeypatch):
    import agent.app as appmod
    from fastapi.testclient import TestClient
    tenant = "11111111-1111-4111-8111-111111111111"
    seen = {}
    monkeypatch.setattr(appmod, "check_auth", lambda r: True)

    def exchange(st, message, sid, owner, now, day, background, extra, emit=None):
        seen["turns"] = [dict(t) for t in appmod.SESSIONS[f"{owner}:{sid}"]["turns"]]
        return {"reply": "Synthetic answer", "decision": "skipped", "xid": None, "reasoning": "", "files": {}}

    monkeypatch.setattr(appmod, "_run_exchange", exchange)
    body = {"message": "NEW MESSAGE", "files": {}, "history": [
        {"role": "user", "content": "STALE QUESTION"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Answered question"},
        {"role": "assistant", "content": "Answered reply"},
    ]}
    r = TestClient(appmod.app).post("/api/local-exchange", headers={"X-Cowork-User-ID": tenant}, json=body)
    assert r.status_code == 200, r.text
    assert seen["turns"] == [{"role": "user", "content": "Answered question"},
                             {"role": "assistant", "content": "Answered reply"}]
    assert not any(key.startswith(f"{tenant}:") for key in appmod.SESSIONS)
