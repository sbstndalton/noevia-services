"""R9 answering (D9): the shadow comparison of the Rust Diary process's answer half.

DIARY_RUST_ANSWER selects it:
- empty (the default): off. Nothing here runs, the Rust process gets no ``answer`` start-up config
  (so its ``/internal/answer`` and ``/internal/reindex`` answer 404) and behaviour is byte-identical
  to a release without this module.
- ``shadow``: for each exchange of a managed tenant, the Rust process (noevia-rs bins/noevia-diary
  answer.rs, ``POST /internal/answer`` with ``mode: "context"``) is asked for the model messages it
  would send, built from the same inputs Python used; the two lists are compared and only the
  outcome is logged and counted (positions, lengths and roles, never content). Python still builds
  the context it sends, answers, saves and holds the tenant writer lock: the shadow changes nothing
  the user sees except, at most, a bounded wait before the save (below).

The Rust side reads the tenant's corpus and index while Python is waiting for its model. Python's
save changes that corpus, so the save waits for the shadow to finish, at most until the shadow's
deadline; a shadow still running then is counted ``late`` and its result is ignored, so a
comparison never reads the state after the save. A shadow request is refused, declined (421: the
Rust side does not reproduce this tenant or state) or failed: each is counted, none affects the
exchange.

Another write of the same tenant landing while the shadow reads (an edit, an import) can still make
a comparison differ; a mismatch is a lead to check, not proof of a fault.

Not compared: the retrieval trace, and an exchange that carries a local reference (the streamed
route's file reference, which the Rust answer half does not build): such an exchange is counted
``skipped``.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable, Optional

import httpx

log = logging.getLogger("diary.rust.answer")

MODES = ("", "shadow")
# One shadow request: connect, the Rust context build (one query embedding plus the index) and the
# reply. The save waits for it at most this long after the shadow started.
SHADOW_DEADLINE_S = 10.0
OUTCOMES = ("match", "mismatch", "declined", "refused", "failed", "late", "skipped")

_stats_lock = threading.Lock()
STATS = {k: 0 for k in OUTCOMES}
_warned = set()


def mode(raw: Optional[str] = None) -> str:
    """DIARY_RUST_ANSWER, lower-cased; an unknown value is off, with one warning."""
    value = (os.environ.get("DIARY_RUST_ANSWER", "") if raw is None else raw).strip().lower()
    if value in MODES:
        return value
    if value not in _warned:
        _warned.add(value)
        log.warning("DIARY_RUST_ANSWER: %r is not one of %s; the answer shadow stays off", value[:32], MODES[1:])
    return ""


def wanted() -> bool:
    return mode() != ""


def answer_config(base_cfg, reasoning_tokens: int) -> Optional[dict]:
    """The ``answer`` start-up config (noevia-diary answer.rs ``AnswerConfig``), None when off."""
    if not wanted():
        return None
    llm = base_cfg.get("llm") or {}
    return {
        "embedding_seed": {"llm": {k: v for k, v in llm.items() if isinstance(k, str) and k.startswith("embed_")}}
        if isinstance(llm, dict) else {"llm": {}},
        "timeout_s": float(base_cfg.get("llm.timeout_s", 300)),
        "max_retries": int(base_cfg.get("llm.max_retries", 3)),
        "reasoning_tokens": int(reasoning_tokens),
    }


def _count(outcome: str) -> None:
    with _stats_lock:
        STATS[outcome] += 1


def stats() -> dict:
    with _stats_lock:
        return dict(STATS)


def _plain(messages) -> list:
    return [{"role": m.get("role"), "content": m.get("content")} for m in messages]


def compare(python: list, rust) -> Optional[str]:
    """None when equal, else where they first differ: positions, roles and lengths, no content."""
    if not isinstance(rust, list):
        return "the Rust reply has no message list"
    py = _plain(python)
    for i, (a, b) in enumerate(zip(py, rust)):
        if not isinstance(b, dict):
            return f"message {i}: not an object"
        if a["role"] != b.get("role"):
            return f"message {i}: role {a['role']!r} vs {str(b.get('role'))[:16]!r}"
        ca, cb = a["content"], b.get("content")
        if ca != cb:
            if not isinstance(cb, str) or not isinstance(ca, str):
                return f"message {i} ({a['role']}): content type differs"
            at = next((j for j, (x, y) in enumerate(zip(ca, cb)) if x != y), min(len(ca), len(cb)))
            return f"message {i} ({a['role']}): content differs at character {at} (lengths {len(ca)} vs {len(cb)})"
    if len(py) != len(rust):
        return f"message counts differ ({len(py)} vs {len(rust)})"
    return None


class Shadow:
    """One shadow comparison, running beside the exchange on its own thread."""

    def __init__(self, base_url: str, token: str, tenant_id: str, request: dict, python_messages: list,
                 transport: Optional[httpx.BaseTransport] = None, deadline_s: Optional[float] = None):
        self._base_url = base_url
        self._token = token
        self._tenant = tenant_id
        self._request = request
        self._python = _plain(python_messages)
        self._transport = transport
        self._deadline_s = SHADOW_DEADLINE_S if deadline_s is None else deadline_s
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._closed = False  # set by finish(): a result arriving afterwards is ignored
        self.outcome: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="rust-answer-shadow", daemon=True)

    def start(self) -> "Shadow":
        self._thread.start()
        return self

    def _settle(self, outcome: str, detail: str = "") -> None:
        with self._lock:
            if self._closed or self.outcome is not None:
                return
            self.outcome = outcome
        _count(outcome)
        if outcome == "match":
            log.info("Diary answer shadow: context matches (%d messages)", len(self._python))
        elif outcome == "mismatch":
            log.warning("Diary answer shadow: context differs: %s", detail)
        else:
            log.info("Diary answer shadow: %s%s", outcome, f" ({detail})" if detail else "")

    def _run(self) -> None:
        from .rust_forward import CONTROL_HEADER, DECLINE_HEADER
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._deadline_s, trust_env=False,
                              transport=self._transport) as client:
                resp = client.post("/internal/answer", params={"tenant": self._tenant}, json=self._request,
                                   headers={CONTROL_HEADER: self._token})
        except httpx.HTTPError as exc:
            self._settle("failed", type(exc).__name__)
            return
        if resp.status_code == 421 and resp.headers.get(DECLINE_HEADER.decode()) == "1":
            self._settle("declined")
            return
        if resp.status_code != 200:
            self._settle("refused", f"HTTP {resp.status_code}")
            return
        try:
            rust = resp.json().get("messages")
        except (ValueError, AttributeError):
            self._settle("failed", "unreadable reply")
            return
        difference = compare(self._python, rust)
        self._settle("mismatch" if difference else "match", difference or "")

    def finish(self) -> Optional[str]:
        """Before the save: wait for the comparison, at most until its deadline. A shadow still running
        then is counted late and whatever it finds later is ignored."""
        remaining = self._deadline_s - (time.monotonic() - self._started)
        if remaining > 0:
            self._thread.join(remaining)
        with self._lock:
            self._closed = True
            late = self.outcome is None
        if late:
            _count("late")
            log.info("Diary answer shadow: late (no reply before the save)")
            return "late"
        return self.outcome


def begin(st, tenant_id: str, message: str, turns: list, day, extra_context: str, messages: list,
          max_answer_tokens: int, server_lookup: Optional[Callable[[], object]] = None,
          transport: Optional[httpx.BaseTransport] = None) -> Optional[Shadow]:
    """Start the shadow comparison for one exchange, or None when it does not apply (off, legacy
    tenant, no ready Rust process) or is skipped. Never raises."""
    if not wanted() or tenant_id == "legacy":
        return None
    try:
        if server_lookup is None:
            from . import rust_forward
            server_lookup = rust_forward.current
        server = server_lookup()
        if server is None:
            return None
        token, ready = server.control()
        if not ready or not token:
            return None
        if getattr(st.assembler, "local_reference", ""):
            _count("skipped")
            log.info("Diary answer shadow: skipped (local reference)")
            return None
        config = getattr(getattr(st.retrieval, "embedding", None), "config", None)
        if config is None:
            _count("skipped")
            log.info("Diary answer shadow: skipped (no shared embedding configuration)")
            return None
        cfg = st.assembler.cfg
        request = {
            "mode": "context",
            "message": message,
            "turns": [dict(t) for t in turns],
            "day": day.isoformat(),
            "extra_context": extra_context,
            "trace": False,
            "max_answer_tokens": int(max_answer_tokens),
            "cfg": {"context": cfg.get("context") or {}, "retrieval": cfg.get("retrieval") or {}},
            "system_prompt": st.assembler._system_prompt(),
            "stream": False,
            "embedding": config.get(),
        }
        json.dumps(request)  # a config value JSON cannot carry: skipped, not failed later
    except (TypeError, ValueError, OSError) as exc:
        _count("skipped")
        log.info("Diary answer shadow: skipped (%s)", type(exc).__name__)
        return None
    except Exception:  # the shadow must never fail an exchange
        log.exception("Diary answer shadow: could not start")
        _count("failed")
        return None
    return Shadow(server.base_url, token, tenant_id, request, messages, transport=transport).start()
