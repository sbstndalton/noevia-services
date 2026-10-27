"""#1505: a model's reasoning is never shown to the user and never saved. Synthetic fixtures only."""
import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from agent.llm import ChatReply, LLMClient, LLMError
from agent.streaming import exchange_stream


def _sse(chunks):
    body = "".join("data: " + json.dumps({"choices": [{"delta": d}]}) + "\n\n" for d in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})


def _client(handler):
    client = LLMClient("http://synthetic.invalid/v1", chat_model="synthetic-model")
    client._client.close()
    client._client = httpx.Client(base_url="http://synthetic.invalid/v1", transport=httpx.MockTransport(handler))
    return client


THINK_CHUNKS = [
    {"content": "<thi"}, {"content": "nk>Synthetic secret plan"}, {"content": " about the reply</thi"},
    {"content": "nk>Visible answer."}, {"content": " More of it.\n[LOG: ok]"},
]


def test_think_blocks_are_not_streamed_and_not_returned_as_answer_text():
    client = _client(lambda request: _sse(THINK_CHUNKS))
    events = []
    reply = client.chat_stream([], events.append)
    shown = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "secret" not in shown and "<think" not in shown
    answer = [e for e in events if e["type"] == "answer"]
    assert answer and answer[0]["text"] == "Visible answer. More of it."
    assert "secret" not in reply and reply.startswith("Visible answer.")
    client.close()


def test_provider_reasoning_deltas_are_not_streamed_to_the_user():
    client = _client(lambda request: _sse([{"reasoning_content": "Synthetic live thinking"}, {"content": "Visible answer.\n[LOG: ok]"}]))
    events = []
    client.chat_stream([], events.append)
    assert not any(e["type"] == "reasoning" for e in events)
    assert not any("Synthetic live thinking" in json.dumps(e) for e in events)
    client.close()


def test_an_unclosed_think_block_is_not_an_answer():
    client = _client(lambda request: _sse([{"content": "<think>Synthetic reasoning that never closes"}]))
    events = []
    with pytest.raises(LLMError, match="no final answer"):
        client.chat_stream([], events.append)
    assert not any(e["type"] in ("delta", "answer") for e in events)
    client.close()


def test_non_streamed_reply_strips_think_blocks_from_content():
    client = _client(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "<think>Synthetic secret</think>Visible answer."}}]}))
    reply = client.chat([])
    assert reply == "Visible answer."
    assert "secret" not in reply
    client.close()


def test_the_relay_never_forwards_a_reasoning_event_to_the_browser():
    def work(emit):
        emit({"type": "status", "text": "Generating companion response…"})
        emit({"type": "reasoning", "text": "Synthetic secret thought"})
        emit({"type": "answer", "text": "Visible answer."})
        return {"reply": "Visible answer.", "decision": "skipped"}

    async def run():
        return [c async for c in exchange_stream(work).body_iterator]

    wire = "".join(asyncio.run(run()))
    assert "Synthetic secret thought" not in wire and '"reasoning"' not in wire
    assert "Visible answer." in wire


def test_saved_entry_and_stream_carry_no_reasoning_from_a_real_exchange(monkeypatch):
    import agent.app as appmod
    from agent.pipeline import LoggingPipeline
    from agent.deadline import ExchangeBudget  # noqa: F401 - the exchange builds its own budget
    client = _client(lambda request: _sse(THINK_CHUNKS))
    saved = []

    class Store:
        def require_write_transform(self):
            pass  # This test double represents an available Python transform.

        index_enabled = False

        def get_standing_sections_text(self, max_chars=0):
            return ""

        def log_exchange(self, **kwargs):
            saved.append(kwargs)
            return "xid-synthetic"

    class Aux:
        def chat(self, messages, **kwargs):
            if messages[-1]["content"].startswith("SUMMARIZE"):
                raise RuntimeError("synthetic summarizer down")  # forces the verbatim fallback, the riskier save path
            return "LOG"

    templates = {"skip_classifier": "CLASSIFY {user_message} {assistant_message}", "summarizer": "SUMMARIZE {assistant_message}"}
    store = Store()
    st = SimpleNamespace(
        assembler=SimpleNamespace(build=lambda day, message, session_turns=None: [{"role": "user", "content": message}]),
        llm_main=client,
        pipeline=LoggingPipeline(store, client, Aux(), templates),
        store=store,
    )
    events = []
    monkeypatch.setattr(appmod, "_reindex_dirty", lambda s: None)
    try:
        result = appmod._run_exchange(st, "Synthetic question", "reasoning-saved", "legacy", background=False, emit=events.append)
    finally:
        appmod.SESSIONS.pop("legacy:reasoning-saved", None)
        client.close()
    assert result["decision"] == "logged" and result["reasoning"] == ""
    assert saved and "secret" not in json.dumps(saved, ensure_ascii=False, default=str)
    assert "secret" not in json.dumps(events, ensure_ascii=False)
    assert "secret" not in json.dumps(result, ensure_ascii=False)


# ---- review follow-up (#1505): only leading reasoning is removed -----------------------------------------


@pytest.mark.parametrize("text", [
    "Use the <think> tag to start a reasoning block.",
    "Here is the snippet: <think>not reasoning</think> and more.",
    "Answer first.\n<thinking>an aside the user asked for</thinking>",
    "A lone <thi at the end is just text <thi",
])
def test_a_literal_think_tag_inside_a_reply_is_kept(text):
    from agent.llm import strip_reasoning
    assert strip_reasoning(text) == text
    assert strip_reasoning(text, final=False) == text


@pytest.mark.parametrize("text,expected", [
    ("<think>Synthetic secret</think>Visible answer.", "Visible answer."),
    ("\n  <THINK>Synthetic secret</think>\n\nVisible answer.", "Visible answer."),
    ("<think>one</think><thinking>two</thinking>Visible answer. <think>kept</think>", "Visible answer. <think>kept</think>"),
    ("<think>Synthetic reasoning that never closes", ""),
    ("<think>Visible-looking text after an opener</thin", ""),
])
def test_leading_reasoning_is_removed(text, expected):
    from agent.llm import strip_reasoning
    assert strip_reasoning(text) == expected


def test_a_streamed_prefix_never_shrinks_whatever_the_chunking():
    from agent.llm import strip_reasoning
    for full in ["<think>Synthetic secret plan</think>Visible answer here.", "Use the <think> tag like so.",
                 "  \n<thinking>plan</thinking>\nAnswer."]:
        shown_before = ""
        for end in range(len(full) + 1):
            shown = strip_reasoning(full[:end], final=False)
            assert shown.startswith(shown_before), (full[:end], shown_before, shown)
            shown_before = shown


def test_a_reply_that_mentions_the_tag_mid_text_is_streamed_whole():
    chunks = [{"content": "You can write "}, {"content": "<think>"}, {"content": " yourself if you like.\n[LOG: ok]"}]
    client = _client(lambda request: _sse(chunks))
    events = []
    reply = client.chat_stream([], events.append)
    assert reply.startswith("You can write <think> yourself if you like.")
    answer = [e for e in events if e["type"] == "answer"]
    assert answer and answer[0]["text"] == "You can write <think> yourself if you like."
    client.close()
