"""LLM client for OpenAI-compatible chat and embedding endpoints."""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import re
import socket
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx

from .util import ensure_not_redirect, make_client

log = logging.getLogger(__name__)

_LOG_MARKER_RE = re.compile(r"\n?\s*\[LOG:\s*(ok|skip)\s*\]\s*$")
# #1505: a model that writes its reasoning inline puts it FIRST: one or more complete leading blocks are
# removed, and a leading opener with no closer means the reasoning never ended, so nothing follows it as an
# answer. A literal "<think>" later in a reply (a question about the tag, a quoted snippet) is the user's
# text and stays.
_THINK_LEAD_BLOCK_RE = re.compile(r"\s*<(think|thinking)>.*?</\1>\s*", re.DOTALL | re.IGNORECASE)
_THINK_LEAD_OPEN_RE = re.compile(r"\s*<(think|thinking)>", re.IGNORECASE)
_THINK_OPENERS = ("<think>", "<thinking>")


def strip_reasoning(text: str, final: bool = True) -> str:
    """Visible text without a model's leading inline reasoning. With final=False (while streaming) the
    text is held back while it could still turn into a leading opener (nothing but whitespace, or a
    prefix of one), so a streamed prefix never shrinks."""
    while True:
        block = _THINK_LEAD_BLOCK_RE.match(text)
        if not block:
            break
        text = text[block.end():]
    if _THINK_LEAD_OPEN_RE.match(text):
        return ""
    if not final:
        head = text.lstrip().lower()
        if not head or (len(head) < len("<thinking>") and any(opener.startswith(head) for opener in _THINK_OPENERS)):
            return ""
    return text


# #127: the default answer-length cap (tokens) of the companion answer; llm.max_answer_tokens overrides it.
DEFAULT_MAX_ANSWER_TOKENS = 1024
# #164: the extra tokens a model that streams reasoning may spend before its answer. The model call is sent
# max_tokens = answer cap + this allowance, and the answer cap is then enforced on the visible answer only, so
# private reasoning never uses up the answer. llm.max_reasoning_tokens overrides it; 0 restores the #127 request.
DEFAULT_MAX_REASONING_TOKENS = 4096
# #127: appended to an answer the model server cut off at the cap, so a shortened answer is never silent.
SHORTENED_NOTE = "(This answer was shortened: it reached the answer length limit.)"


def with_shortened_note(text: str) -> str:
    return (text.rstrip() + "\n\n" + SHORTENED_NOTE) if text.strip() else SHORTENED_NOTE


class ChatReply(str):
    """Answer text with optional provider reasoning, never part of stored prose. `truncated` is True when
    the provider stopped at the length cap (finish_reason "length", #127)."""
    def __new__(cls, content: str, reasoning: str = "", truncated: bool = False):
        reply = super().__new__(cls, content)
        reply.reasoning = reasoning
        reply.truncated = truncated
        return reply


def _shutdown_upstream(holder: Dict[str, Any]) -> None:
    """#127: close the connection of a given-up model call so the model server notices the disconnect and
    stops generating. A socket shutdown also wakes the thread still blocked reading it; with no socket (an
    in-process transport) the response is closed instead."""
    response = holder.get("response")
    stream = holder.get("network") or (response.extensions.get("network_stream") if response is not None else None)
    sock = None
    if stream is not None:
        try:
            sock = stream.get_extra_info("socket")
        except Exception:  # noqa: BLE001
            sock = None
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        return
    if response is not None:
        response.close()


class LLMError(RuntimeError):
    pass


# #135: the reason code and text for an answer the cap cut off before any answer text (reasoning used it all).
LENGTH_CAP_TEXT = ("The answer cap (LLM_MAX_ANSWER_TOKENS) was reached before any answer text; "
                   "nothing was saved. You can resend or raise the cap (or the reasoning allowance, "
                   "LLM_MAX_REASONING_TOKENS).")


class LengthCapError(LLMError):
    """#135: finish_reason "length" with an empty visible answer. Nothing was saved. Told apart from other
    model failures so /v1 and the stream can report reason "length_cap"."""
    def __init__(self, message: str = LENGTH_CAP_TEXT):
        super().__init__(message)


# #146: the text for an answer that stopped normally but held no answer text (empty, or only the log marker).
EMPTY_ANSWER_TEXT = "The model returned no final answer — nothing was saved. You can resend."


class EmptyAnswerError(LLMError):
    """#146: an empty answer after a normal stop (or a reply holding only the log marker). Nothing was saved.
    Told apart from other model failures, and from LengthCapError (the cap is not why it is empty), so /v1,
    the local exchange and the stream can report reason "empty_answer" with saved false."""
    def __init__(self, message: str = EMPTY_ANSWER_TEXT):
        super().__init__(message)


class _Redirected(LLMError):
    """An embeddings server answered with a redirect; never retried."""


class InputTooLarge(LLMError):
    """#1759: the embeddings server refused an input as longer than it can take in one batch or
    context (llama.cpp: "input (N tokens) is too large to process. increase the physical batch size
    (current batch size: B)", or the context-size form). Never retried: the answer would be the same.
    ``tokens`` and ``limit`` are what the server reported, when it did."""

    def __init__(self, message: str, tokens: Optional[int] = None, limit: Optional[int] = None):
        super().__init__(message)
        self.tokens = tokens
        self.limit = limit


_TOO_LARGE_RE = re.compile(r"too large to process|exceeds the available context size|larger than the max context size"
                           r"|exceed_context_size", re.IGNORECASE)
_TOO_LARGE_TOKENS_RE = re.compile(r"input \((\d+) tokens\)|\"n_prompt_tokens\"\s*:\s*(\d+)")
_TOO_LARGE_LIMIT_RE = re.compile(r"current batch size:\s*(\d+)|\"n_ctx\"\s*:\s*(\d+)")


def _first_int(rx: "re.Pattern[str]", text: str) -> Optional[int]:
    m = rx.search(text)
    if not m:
        return None
    return int(next(g for g in m.groups() if g))


def too_large_from_response(resp: httpx.Response) -> Optional[InputTooLarge]:
    """An InputTooLarge for an error answer that says the input is too long; None for any other answer."""
    if resp.status_code < 400:
        return None
    try:
        body = resp.text[:4000]
    except Exception:  # noqa: BLE001
        return None
    if not _TOO_LARGE_RE.search(body):
        return None
    return InputTooLarge(f"embeddings refused as too long (HTTP {resp.status_code}): {body[:300]}",
                         _first_int(_TOO_LARGE_TOKENS_RE, body), _first_int(_TOO_LARGE_LIMIT_RE, body))


# #1422 / #1065: the model the companion is configured for need not be an id the local engine
# serves. The served list is read at call time and remembered this long.
SERVED_TTL_S = 60.0
_LIST_TIMEOUT_S = 3.0
# Router launch flags that make a section something other than a chat model. Read from the router's
# own per-model status (flags, never names).
_NON_CHAT_FLAGS = frozenset({"--embedding", "--embeddings", "--reranking", "--rerank"})


# #112: the listing scope of one GET /api/health check (per thread; None outside a check).
_listing_scope = threading.local()


@contextmanager
def one_listing_per_check() -> Iterator[None]:
    """#112: inside this block each engine (base URL and credential) is listed at most once. Every
    client that asks gets the same answer, a failure included, so the chat and aux clients on one
    engine cost one /models call per health check. A nested block shares the outer block answers."""
    if getattr(_listing_scope, "memo", None) is not None:
        yield
        return
    _listing_scope.memo = {}
    try:
        yield
    finally:
        _listing_scope.memo = None


class EngineUnreachable(LLMError):
    """The model engine could not be contacted at all (transport failure)."""


class ModelNotServed(LLMError):
    """#2099: no model id the engine serves is configured for this role. The request is refused
    before anything is sent, so the engine never loads (and evicts for) a model nobody chose.
    `setting` names the configuration key; the message carries no model id."""

    def __init__(self, setting: str, message: str):
        super().__init__(message)
        self.setting = setting


def _served_entries(data: Any) -> List[Dict[str, Any]]:
    """The router's /v1/models body as [{id, loaded, chat}]. A fixed single-model server has no
    per-model `status`; it only answers once that model is loaded."""
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("unexpected model list shape")
    out = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            continue
        status = item.get("status")
        args = status.get("args") if isinstance(status, dict) else None
        out.append({
            "id": item["id"],
            "loaded": not isinstance(status, dict) or status.get("value") == "loaded",
            "chat": not (isinstance(args, list) and _NON_CHAT_FLAGS.intersection(a for a in args if isinstance(a, str))),
        })
    return out


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        chat_model: str = "",
        embed_model: str = "",
        timeout_s: float = 300.0,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        embed_base_url: str = "",
        resolve_served_model: bool = False,
        prefer_model_from: Optional["LLMClient"] = None,
        fallback_model: str = "",
        setting_key: str = "LLM_CHAT_MODEL",
        fallback_setting_key: str = "LLM_CHAT_FALLBACK_MODEL",
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        # Opt-in so a client built for a fixed endpoint behaves exactly as before. When on, every
        # chat call uses a model the endpoint serves (see resolve_chat_model).
        self.resolve_served_model = resolve_served_model
        self._prefer_from = prefer_model_from
        # #2099: the only substitute ever used is one the operator named; never "first listed".
        self.fallback_model = (fallback_model or "").strip()
        self.setting_key = setting_key
        self.fallback_setting_key = fallback_setting_key
        self._resolve_lock = threading.Lock()
        self._resolved: Optional[Tuple[float, str, str]] = None  # (expires, model, source)
        self._logged: Optional[Tuple[str, str]] = None
        self.embed_model = embed_model
        self.max_retries = max_retries
        # #164: added to every capped request; the cap itself is enforced on the visible answer.
        self.reasoning_tokens = DEFAULT_MAX_REASONING_TOKENS
        self.retry_backoff_s = retry_backoff_s
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # Operator-configured endpoint (not user-supplied), so redirects stay allowed.
        self._client = make_client(base_url=self.base_url, timeout_s=timeout_s, headers=headers, follow_redirects=True)
        # #697: embeddings may have their own server (the CPU `embed` sidecar), so they never ask
        # the one-model inference engine to swap its chat model out. Like web's EMBEDDING_BASE_URL,
        # the inference credential is not sent there.
        embed_base = (embed_base_url or "").rstrip("/")
        # Embedding calls never follow a redirect, on either path: the administrator can point them at
        # another address at runtime (embedding_migration), and a 307/308 would carry the Diary's text
        # wherever it says. The chat client above stays as it was.
        self._embed_client = (
            make_client(base_url=embed_base, timeout_s=timeout_s, headers={"Content-Type": "application/json"}, follow_redirects=False)
            if embed_base and embed_base != self.base_url
            else make_client(base_url=self.base_url, timeout_s=timeout_s, headers=headers, follow_redirects=False)
        )
    # ---------------- served-model resolution ----------------

    def _list_served(self) -> List[Dict[str, Any]]:
        # #112: inside one_listing_per_check, clients that reach the same endpoint with the same
        # credential share one listing, a failed one included.
        memo = getattr(_listing_scope, "memo", None)
        key = (self.base_url, self._client.headers.get("Authorization", ""))
        if memo is not None and key in memo:
            shared = memo[key]
            if isinstance(shared, Exception):
                raise shared
            return shared
        try:
            entries = self._fetch_served()
        except Exception as exc:
            if memo is not None:
                memo[key] = exc
            raise
        if memo is not None:
            memo[key] = entries
        return entries

    def _fetch_served(self) -> List[Dict[str, Any]]:
        try:
            resp = self._client.get("/models", timeout=_LIST_TIMEOUT_S)
        except httpx.TransportError as exc:
            raise EngineUnreachable(f"model engine unreachable ({type(exc).__name__})") from exc
        resp.raise_for_status()
        return _served_entries(resp.json())

    def resolve_chat_model(self) -> Tuple[str, str]:
        """(model id, source) to send as `model`. Source is "configured" (the endpoint serves the
        configured id), "fallback" (it does not, so the explicitly configured fallback id, or the id
        the partner client resolved to, is used; both are served) or "unverified" (the endpoint cannot
        list models, so the configured id is kept). #2099: a request never goes out without an
        explicit id the endpoint serves. When neither the configured id nor a configured substitute is
        served, raises ModelNotServed (naming the setting key, not its value) and sends nothing; the
        engine is never left to pick a model. Raises EngineUnreachable when the engine cannot be
        contacted and LLMError when it serves no chat model. The answer is remembered for
        SERVED_TTL_S (a refusal is not). Never leaves the configured endpoint."""
        if not self.resolve_served_model:
            if not self.chat_model.strip():
                raise self._not_configured()
            return self.chat_model, "configured"
        with self._resolve_lock:
            now = time.monotonic()
            if self._resolved and self._resolved[0] > now:
                return self._resolved[1], self._resolved[2]
            try:
                entries = self._list_served()
            except EngineUnreachable:
                self._resolved = None
                log.error("Diary model engine could not be reached to list its models")
                raise
            except Exception as exc:  # noqa: BLE001 - endpoint without a usable model list
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if not self.chat_model.strip():
                    self._resolved = None
                    raise self._not_configured() from exc
                log.warning("model list unavailable (%s%s); using the model set in %s",
                            type(exc).__name__, f" {status}" if status else "", self.setting_key)
                model, source, why = self.chat_model, "unverified", "the endpoint did not list its models"
            else:
                try:
                    model, source, why = self._choose(entries)
                except ModelNotServed as refusal:
                    self._resolved = None
                    if self._logged != ("", "not_served"):
                        self._logged = ("", "not_served")
                        log.error("%s", refusal)
                    raise
            self._resolved = (now + SERVED_TTL_S, model, source)
            if self._logged != (model, source):
                self._logged = (model, source)
                if source == "fallback":
                    log.warning("Diary chat model: the id set in %s is not served; using %s", self.setting_key, why)
                else:
                    log.info("Diary chat model chosen (%s): %s", source, why)
            return model, source

    def _not_configured(self) -> ModelNotServed:
        return ModelNotServed(
            self.setting_key,
            f"no model id is set in {self.setting_key}; set it to an id the model engine serves "
            "(the diary never lets the engine choose a model)")

    def _choose(self, entries: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        # #112: the configured id must be a chat-capable served id (not an embedding or reranking
        # section), not merely a listed one.
        chat_ids = {e["id"] for e in entries if e["chat"]}
        if not chat_ids:
            raise LLMError("the model engine serves no chat model")
        if self.chat_model in chat_ids:
            return self.chat_model, "configured", "the configured model is served"
        if self.fallback_model and self.fallback_model in chat_ids:
            return self.fallback_model, "fallback", f"the explicit fallback set in {self.fallback_setting_key}"
        # #112: the last substitute of the aux role is the resolved id of the chat client. The caller
        # sets prefer_model_from only when both roles name the same engine. This is not an explicit
        # fallback: the operator named no id for aux here, and the chat id is whatever chat resolved
        # to, including the chat fallback. It is used only after the aux id and the aux fallback are
        # both unserved.
        partner = ""
        if self._prefer_from is not None and self._prefer_from is not self:
            try:
                partner = self._prefer_from.resolve_chat_model()[0]
            except LLMError:
                pass
        if partner and partner in chat_ids:
            return partner, "fallback", "the chat model's id"
        raise ModelNotServed(
            self.setting_key,
            f"the model id set in {self.setting_key} is not served by the model engine; "
            f"set {self.setting_key} to a served id"
            + (f" (the fallback set in {self.fallback_setting_key} is not served either)" if self.fallback_model else
               f" or set {self.fallback_setting_key} to a served id")
            + ". No request was sent.")

    def invalidate_resolution(self) -> None:
        with self._resolve_lock:
            self._resolved = None

    # ---------------- shared retry plumbing ----------------

    def _post_with_retries(self, path: str, payload: Dict[str, Any], describe: str, parse, client=None,
                           refuse_redirect: bool = False, detect_too_large: bool = False) -> Any:
        """POST with bounded retries; `parse` validates the response body and
        returns the result. Shared by chat() and embed(), which previously
        duplicated the retry loop (including its bug of sleeping after the
        final failed attempt before raising)."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = (client or self._client).post(path, json=payload)
                if refuse_redirect:
                    try:
                        ensure_not_redirect(resp)  # raise_for_status() counts a 3xx as success
                    except RuntimeError as redirected:
                        raise _Redirected(f"{describe} server redirected (HTTP {resp.status_code}); redirects are refused") from redirected
                if detect_too_large:
                    too_large = too_large_from_response(resp)
                    if too_large is not None:
                        raise too_large
                resp.raise_for_status()
                return parse(resp.json())
            except (_Redirected, InputTooLarge):
                raise  # not retried: the answer would be the same, and each try would send the text again
            except Exception as exc:  # noqa: BLE001 - retry any transport/parse failure
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break  # last attempt: no point sleeping before raising
                wait = self.retry_backoff_s * (2**attempt)
                log.warning("%s attempt %d failed (%s); retrying in %.1fs", describe, attempt + 1, exc, wait)
                time.sleep(wait)
        raise LLMError(f"{describe} failed after {self.max_retries} attempts: {last_exc}") from last_exc

    # ---------------- chat ----------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        alive=None,
    ) -> str:
        """One chat completion. With `alive` (the exchange deadline hook, #127) the request is streamed
        internally and its connection is closed when the deadline gives the call up, so the model server
        stops generating; the reply is the same as the non-streamed one."""
        if alive is not None:
            try:
                return self._chat_cancellable(messages, model, temperature, max_tokens, stop, alive)
            except LLMError:
                self.invalidate_resolution()  # the served list may have changed; look again next time
                raise
        payload: Dict[str, Any] = {
            "model": model or self.resolve_chat_model()[0],
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = self._request_tokens(max_tokens)
        if stop:
            payload["stop"] = stop

        def parse_chat(data: Any) -> str:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise LLMError(f"unexpected chat response shape: {data!r}")
            message = data["choices"][0]["message"]
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            return ChatReply(strip_reasoning(content), reasoning if isinstance(reasoning, str) else "",
                             data["choices"][0].get("finish_reason") == "length")

        try:
            return self._post_with_retries("/chat/completions", payload, "chat", parse_chat)
        except LLMError:
            self.invalidate_resolution()  # the served list may have changed; look again next time
            raise

    def _chat_cancellable(self, messages, model, temperature, max_tokens, stop, alive):
        """#127: chat() for a call bounded by the exchange deadline. Streamed internally (a model server
        notices a client disconnect while it streams), accumulated, and closed when the call is given up.
        A failure before the provider answered (no connection, a 5xx) is retried as chat() does; once the
        answer has begun, or after the call was given up, nothing is sent again."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            if getattr(alive, "abandoned", False):
                break
            try:
                content, reasoning, finish = self._stream_completion(messages, model, temperature, max_tokens, stop, alive, None)
                return ChatReply(strip_reasoning(content), reasoning, finish == "length")
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.HTTPStatusError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", 500)
                if status < 500:
                    raise LLMError(f"chat failed: HTTP {status}") from exc
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                wait = self.retry_backoff_s * (2**attempt)
                log.warning("chat attempt %d failed (%s); retrying in %.1fs", attempt + 1, type(exc).__name__, wait)
                time.sleep(wait)
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport/parse failure after the answer began
                raise LLMError(f"chat failed: {type(exc).__name__}") from exc
        raise LLMError(f"chat failed after {self.max_retries} attempts: {last_exc}") from last_exc

    def chat_stream(self, messages, emit, temperature=0.7, alive=None, max_tokens=None):
        """Stream real provider output once; retain log markers only for capture. `alive`, when given,
        is called on every line the provider sends (#1505: the exchange deadline counts it as progress).
        `max_tokens` caps the answer (#127); an answer cut off there ends with SHORTENED_NOTE."""
        try:
            return self._chat_stream(messages, emit, temperature, alive, max_tokens)
        except Exception:
            self.invalidate_resolution()  # the served list may have changed; look again next time
            raise

    def _chat_stream(self, messages, emit, temperature, alive=None, max_tokens=None):
        content, reasoning, finish = self._stream_completion(messages, None, temperature, max_tokens, None, alive, emit)
        content = strip_reasoning(content)  # #1505: the final answer never carries inline reasoning
        visible, _ = self.strip_log_marker(content)
        truncated = finish == "length"
        if not visible and truncated:
            raise LengthCapError()  # #135: the cap, not the provider, is why there is no answer
        if not visible:
            raise EmptyAnswerError()  # #146: a normal stop with no answer text
        emit({"type": "answer", "text": with_shortened_note(visible) if truncated else visible})
        return ChatReply(content, reasoning, truncated)

    def _request_tokens(self, answer_cap: int) -> int:
        """#164: the max_tokens sent to the model server: the visible-answer cap plus the reasoning allowance."""
        try:
            allowance = max(0, int(getattr(self, "reasoning_tokens", 0) or 0))
        except (TypeError, ValueError):
            allowance = 0
        return int(answer_cap) + allowance

    def _stream_completion(self, messages, model, temperature, max_tokens, stop, alive, emit):
        """One streamed request: (content, reasoning, finish_reason). `emit` (optional) receives deltas and
        telemetry. #127: when `alive` offers on_abandon, the connection is shut down as soon as the
        exchange deadline gives the call up, also while waiting for the first byte."""
        chat_model = model or self.resolve_chat_model()[0]
        payload: Dict[str, Any] = {"model": chat_model, "messages": messages,
                                   "temperature": temperature, "stream": True}
        if max_tokens is not None:
            payload["max_tokens"] = self._request_tokens(max_tokens)
        if stop:
            payload["stop"] = stop

        def telemetry(data):
            if emit is None:
                return
            timings = data.get("timings")
            if isinstance(timings, dict) and "draft_n" in timings and "draft_n_accepted" in timings:
                emit({"type": "mtp", "model": data.get("model") or chat_model,
                      "timings": {"draft_n": timings["draft_n"], "draft_n_accepted": timings["draft_n_accepted"]}})
        holder: Dict[str, Any] = {}

        def trace(name, info):
            # The connection, as soon as it exists: a reply that has not started yet can still be closed.
            if name == "connection.connect_tcp.complete" and isinstance(info, dict):
                holder["network"] = info.get("return_value")

        on_abandon = getattr(alive, "on_abandon", None)
        unregister = on_abandon(lambda: _shutdown_upstream(holder)) if callable(on_abandon) else None
        content, reasoning, sent, finish = "", "", 0, None
        completed = False
        # #164: the answer cap counts visible answer pieces only (one streamed content piece is one token on an
        # OpenAI-compatible server); reasoning, separate or inline, is not counted. Past the cap the stream is
        # closed, which stops the model server, and the answer ends as a length cut.
        visible_pieces, visible_len, capped = 0, 0, False
        try:
            with self._client.stream("POST", "/chat/completions", json=payload, extensions={"trace": trace}) as response:
                holder["response"] = response
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    data = json.loads(response.read())
                    if alive is not None:
                        alive()
                    telemetry(data)
                    completed = True
                    choice = data["choices"][0]
                    finish = choice.get("finish_reason")
                    message = choice["message"]
                    content = message.get("content") or ""
                    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                    if not isinstance(content, str) or not isinstance(reasoning, str):
                        raise LLMError("invalid provider answer")
                    # #1505: provider reasoning is kept request-local, never streamed or stored as the answer.
                    content = strip_reasoning(content)
                else:
                    for line in response.iter_lines():
                        if alive is not None:
                            alive()
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            completed = True
                            break
                        data = json.loads(raw)
                        telemetry(data)
                        if data.get("error"):
                            raise LLMError("provider stream failed")
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        if choices[0].get("finish_reason"):
                            completed = True
                            finish = choices[0]["finish_reason"]
                        delta = choices[0].get("delta") or {}
                        thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if isinstance(thought, str) and thought:
                            reasoning += thought  # #1505: retained request-local only; never streamed
                        text = delta.get("content") or ""
                        if isinstance(text, str) and text and max_tokens is not None:
                            grown = len(strip_reasoning(content + text, final=False))
                            if grown > visible_len:
                                if visible_pieces >= max_tokens:
                                    completed, finish, capped = True, "length", True
                                    break  # the piece past the cap is dropped; leaving the block closes the stream
                                visible_pieces += 1
                                visible_len = grown
                        if isinstance(text, str):
                            content += text
                            if emit is None:
                                continue
                            # Keep inline reasoning and the trailing protocol marker out of visible tokens.
                            shown = strip_reasoning(content, final=False)
                            end = max(sent, len(shown) - 128)
                            if end > sent:
                                emit({"type": "delta", "text": shown[sent:end]})
                                sent = end
        finally:
            if unregister is not None:
                unregister()
        if not completed:
            raise LLMError("provider stream ended before completion")
        if finish == "length":
            # #127: the cap is logged, never the text. #164: says which limit stopped it.
            if capped:
                log.warning("Diary model answer reached its length cap (%s tokens) and was shortened", max_tokens)
            else:
                log.warning("Diary model call reached its token limit (%s answer + %s reasoning tokens) and was shortened",
                            max_tokens, payload.get("max_tokens", 0) - (max_tokens or 0))
        return content, reasoning, finish

    # ---------------- log-marker handling ----------------

    @staticmethod
    def strip_log_marker(reply: str) -> tuple:
        """Split a reply into (visible_text, decision) where decision is 'ok'|'skip'|None.

        The marker must be the last line; if it appears anywhere else it is treated as
        ordinary text (defensive: keeps prompt-injection markers in diary text harmless).
        """
        m = _LOG_MARKER_RE.search(reply)
        if not m:
            return reply.strip(), None
        return reply[: m.start()].strip(), m.group(1)

    # ---------------- embeddings ----------------

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []

        def parse_embed(data: Any) -> List[List[float]]:
            items = data.get("data", [])
            if len(items) != len(texts):
                raise LLMError(f"embedding count mismatch: sent {len(texts)}, got {len(items)}")
            # API may return embeddings out of order; 'index' restores order.
            items = sorted(items, key=lambda d: d.get("index", 0))
            return [item["embedding"] for item in items]

        return self._post_with_retries(
            "/embeddings",
            {"model": model or self.embed_model, "input": texts},
            "embeddings",
            parse_embed,
            client=self._embed_client,
            refuse_redirect=True,
            detect_too_large=True,
        )

    def close(self) -> None:
        self._embed_client.close()
        self._client.close()
