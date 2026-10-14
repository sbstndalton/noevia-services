"""LLM client for OpenAI-compatible chat and embedding endpoints."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

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


class ChatReply(str):
    """Answer text with optional provider reasoning, never part of stored prose."""
    def __new__(cls, content: str, reasoning: str = ""):
        reply = super().__new__(cls, content)
        reply.reasoning = reasoning
        return reply


class LLMError(RuntimeError):
    pass


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


class EngineUnreachable(LLMError):
    """The model engine could not be contacted at all (transport failure)."""


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
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        # Opt-in so a client built for a fixed endpoint behaves exactly as before. When on, every
        # chat call uses a model the endpoint serves (see resolve_chat_model).
        self.resolve_served_model = resolve_served_model
        self._prefer_from = prefer_model_from
        self._resolve_lock = threading.Lock()
        self._resolved: Optional[Tuple[float, str, str]] = None  # (expires, model, source)
        self._logged: Optional[Tuple[str, str]] = None
        self.embed_model = embed_model
        self.max_retries = max_retries
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
        try:
            resp = self._client.get("/models", timeout=_LIST_TIMEOUT_S)
        except httpx.TransportError as exc:
            raise EngineUnreachable(f"model engine unreachable ({type(exc).__name__})") from exc
        resp.raise_for_status()
        return _served_entries(resp.json())

    def resolve_chat_model(self) -> Tuple[str, str]:
        """(model id, source) to send as `model`; source is "configured" (the endpoint serves the
        configured id), "fallback" (it does not, so another served id was chosen) or "unverified"
        (the endpoint cannot list models, so the configured id is kept). The configured id when the endpoint serves it;
        otherwise a model the endpoint does serve: the one this client's partner already resolved to
        (aux follows chat), else the first loaded chat model, else the first listed chat model. The
        answer is remembered for SERVED_TTL_S. An endpoint that cannot list models keeps the
        configured id. Raises EngineUnreachable when the engine cannot be
        contacted and LLMError when it serves no chat model. Never leaves the configured endpoint."""
        if not self.resolve_served_model:
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
                log.warning("model list unavailable (%s%s); using the configured model %r",
                            type(exc).__name__, f" {status}" if status else "", self.chat_model)
                model, source, why = self.chat_model, "unverified", "the endpoint did not list its models"
            else:
                model, source, why = self._choose(entries)
            self._resolved = (now + SERVED_TTL_S, model, source)
            if self._logged != (model, source):
                self._logged = (model, source)
                (log.info if source != "fallback" else log.warning)(
                    "Diary chat model %r chosen (%s): %s; configured %r", model, source, why, self.chat_model)
            return model, source

    def _choose(self, entries: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        ids = {e["id"] for e in entries}
        if self.chat_model in ids:
            return self.chat_model, "configured", "the configured model is served"
        chat = [e for e in entries if e["chat"]]
        partner = ""
        if self._prefer_from is not None and self._prefer_from is not self:
            try:
                partner = self._prefer_from.resolve_chat_model()[0]
            except LLMError:
                pass
        if partner and any(e["id"] == partner for e in chat):
            return partner, "fallback", "the configured model is not served; using the chat model's"
        for e in chat:
            if e["loaded"]:
                return e["id"], "fallback", "the configured model is not served; using the first loaded model"
        if chat:
            return chat[0]["id"], "fallback", "the configured model is not served and none is loaded; using the first listed model"
        raise LLMError("the model engine serves no chat model")

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
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.resolve_chat_model()[0],
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        def parse_chat(data: Any) -> str:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise LLMError(f"unexpected chat response shape: {data!r}")
            message = data["choices"][0]["message"]
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            return ChatReply(strip_reasoning(content), reasoning if isinstance(reasoning, str) else "")

        try:
            return self._post_with_retries("/chat/completions", payload, "chat", parse_chat)
        except LLMError:
            self.invalidate_resolution()  # the served list may have changed; look again next time
            raise

    def chat_stream(self, messages, emit, temperature=0.7, alive=None):
        """Stream real provider output once; retain log markers only for capture. `alive`, when given,
        is called on every line the provider sends (#1505: the exchange deadline counts it as progress)."""
        try:
            return self._chat_stream(messages, emit, temperature, alive)
        except Exception:
            self.invalidate_resolution()  # the served list may have changed; look again next time
            raise

    def _chat_stream(self, messages, emit, temperature, alive=None):
        chat_model = self.resolve_chat_model()[0]
        payload = {"model": chat_model, "messages": messages,
                   "temperature": temperature, "stream": True}
        def telemetry(data):
            timings = data.get("timings")
            if isinstance(timings, dict) and "draft_n" in timings and "draft_n_accepted" in timings:
                emit({"type": "mtp", "model": data.get("model") or chat_model,
                      "timings": {"draft_n": timings["draft_n"], "draft_n_accepted": timings["draft_n_accepted"]}})
        content, reasoning, sent = "", "", 0
        completed = False
        with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            if "text/event-stream" not in response.headers.get("content-type", ""):
                data = json.loads(response.read())
                if alive is not None:
                    alive()
                telemetry(data)
                completed = True
                message = data["choices"][0]["message"]
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
                    delta = choices[0].get("delta") or {}
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if isinstance(thought, str) and thought:
                        reasoning += thought  # #1505: retained request-local only; never streamed
                    text = delta.get("content") or ""
                    if isinstance(text, str):
                        content += text
                        # Keep inline reasoning and the trailing protocol marker out of visible tokens.
                        shown = strip_reasoning(content, final=False)
                        end = max(sent, len(shown) - 128)
                        if end > sent:
                            emit({"type": "delta", "text": shown[sent:end]})
                            sent = end
        if not completed:
            raise LLMError("provider stream ended before completion")
        content = strip_reasoning(content)  # #1505: the final answer never carries inline reasoning
        visible, _ = self.strip_log_marker(content)
        if not visible:
            raise LLMError("provider returned no final answer")
        emit({"type": "answer", "text": visible})
        return ChatReply(content, reasoning)

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
