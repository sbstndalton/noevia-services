"""LLM client for OpenAI-compatible chat and embedding endpoints."""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from .util import make_client

log = logging.getLogger(__name__)

_LOG_MARKER_RE = re.compile(r"\n?\s*\[LOG:\s*(ok|skip)\s*\]\s*$")


class LLMError(RuntimeError):
    pass


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
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = make_client(base_url=self.base_url, timeout_s=timeout_s, headers=headers)

    # ---------------- shared retry plumbing ----------------

    def _post_with_retries(self, path: str, payload: Dict[str, Any], describe: str, parse) -> Any:
        """POST with bounded retries; `parse` validates the response body and
        returns the result. Shared by chat() and embed(), which previously
        duplicated the retry loop (including its bug of sleeping after the
        final failed attempt before raising)."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(path, json=payload)
                resp.raise_for_status()
                return parse(resp.json())
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
            "model": model or self.chat_model,
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
            return content

        return self._post_with_retries("/chat/completions", payload, "chat", parse_chat)

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
        )

    def close(self) -> None:
        self._client.close()
