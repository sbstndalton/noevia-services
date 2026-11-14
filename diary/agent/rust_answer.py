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

Only an exchange on the tenant's own managed storage is compared (what the Rust side reads): an
exchange on an in-memory corpus (``POST /api/local-exchange``) or any other storage is skipped.

Not compared: the retrieval trace, and an exchange that carries a local reference (the streamed
route's file reference, which the Rust answer half does not build): such an exchange is counted
``skipped``.

- ``on``: the Rust process builds the context and gets the companion's answer (``mode: "answer"``;
  streamed, its events are relayed to the caller as they come), and Python saves it as before:
  the session, the save, the writer lock and the re-index stay here. Python's exchange deadline
  bounds the whole call and closes it when it passes (the first-token deadline is the Rust side's,
  which sees the model's progress). The Rust side's failures map to the exceptions Python's own
  model call raises, so every route answers as before. Whatever the Rust side does not take (the
  conditions of the shadow above, a decline, or a process that cannot be reached or refuses (4xx)
  before any answer began) is answered by Python itself, and counted. A 5xx is the exchange's
  failure (the Rust side may already have called the model: never a second answer). A call given
  up has its socket shut at once, also while the reply has not started (#127).
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

MODES = ("", "shadow", "on")
# One shadow request: connect, the Rust context build (one query embedding plus the index) and the
# reply. The save waits for it at most this long after the shadow started.
SHADOW_DEADLINE_S = 10.0
OUTCOMES = ("match", "mismatch", "declined", "refused", "failed", "late", "skipped")

# DIARY_RUST_ANSWER=on: who answered. ``rust`` (an answer or a model failure from the Rust side),
# ``declined`` / ``unavailable`` (Python answered: the Rust side declined, or could not be reached or
# refused before any answer began), ``skipped`` (Python answered: not a case the Rust side takes).
ANSWER_OUTCOMES = ("rust", "declined", "unavailable", "skipped")
# Reading the Rust reply: connect and send are short; the answer itself is bounded by the exchange
# deadline (which closes the connection), never by a socket timeout.
ANSWER_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)

_stats_lock = threading.Lock()
STATS = {k: 0 for k in OUTCOMES}
ANSWER_STATS = {k: 0 for k in ANSWER_OUTCOMES}
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


def answering() -> bool:
    """DIARY_RUST_ANSWER=on: the Rust process answers (Python still saves)."""
    return mode() == "on"


# #2612: the Diary configures no Python logging, so without a handler of their own the outcome lines
# below that are logged at INFO ("context matches", "plan matches", "skipped", "Python answers") never
# reached the container log; only WARNING lines did, through Python's last-resort handler.
OUTCOME_LOGGERS = ("diary.rust.answer", "diary.rust.index")


def configure_outcome_logging(stream=None) -> bool:
    """With an R9 switch set (DIARY_RUST_ANSWER or DIARY_RUST_INDEX), give the two outcome loggers a
    stderr handler at INFO, once. Nothing changes when both are off, or when logging is already
    configured (a root handler exists, e.g. under a test runner): those lines then reach it as before.
    The outcome lines carry counts, positions and reasons only, never content."""
    from . import rust_index
    if not wanted() and not rust_index.wanted():
        return False
    if logging.getLogger().handlers:
        return False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.set_name("r9-outcomes")
    added = False
    for name in OUTCOME_LOGGERS:
        logger = logging.getLogger(name)
        if any(h.get_name() == "r9-outcomes" for h in logger.handlers):
            continue
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        added = True
    return added


def answer_config(base_cfg, reasoning_tokens: int) -> Optional[dict]:
    """The ``answer`` start-up config (noevia-diary answer.rs ``AnswerConfig``), None when both R9
    switches are off. The Rust process serves its answer and re-index endpoints only with it."""
    from . import rust_index
    if not wanted() and not rust_index.wanted():
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


def _count_answer(outcome: str) -> None:
    with _stats_lock:
        ANSWER_STATS[outcome] += 1


def answer_stats() -> dict:
    with _stats_lock:
        return dict(ANSWER_STATS)


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
        try:
            self._compare()
        except Exception as exc:  # noqa: BLE001 - counted, type name only (no content in logs)
            self._settle("failed", type(exc).__name__)

    def _compare(self) -> None:
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
        return self.close(late_if_running=True)

    def close(self, late_if_running: bool = False) -> Optional[str]:
        """End the comparison now (an exchange that failed before its save: whatever it finds later
        is ignored, uncounted)."""
        with self._lock:
            already = self._closed
            self._closed = True
            late = self.outcome is None
        if late and late_if_running and not already:
            _count("late")
            log.info("Diary answer shadow: late (no reply before the save)")
            return "late"
        return self.outcome


class Snapshot:
    """The inputs both sides build from, taken once before Python's build."""

    def __init__(self, turns: list, embedding):
        self.turns = turns
        self.embedding = embedding


def snapshot(st, turns: list) -> Optional[Snapshot]:
    """None when the shadow is off (Python then builds from the live session turns, as before)."""
    if not wanted():
        return None
    try:
        config = getattr(getattr(st.retrieval, "embedding", None), "config", None)
        return Snapshot([dict(t) for t in list(turns)], config.get() if config is not None else None)
    except Exception as exc:  # noqa: BLE001 - the shadow never fails an exchange
        log.info("Diary answer shadow: no snapshot (%s)", type(exc).__name__)
        return None


def finish(shadow: Optional[Shadow]) -> None:
    if shadow is not None:
        shadow.finish()


def close(shadow: Optional[Shadow]) -> None:
    if shadow is not None:
        shadow.close()


def _managed_state_of(st, tenant_id: str) -> bool:
    """The exchange runs on the tenant's own managed state (what the Rust side reads), not on an
    in-memory corpus (POST /api/local-exchange) or another storage."""
    from .managed_storage import ManagedCorpusBackend
    backend = getattr(st, "backend", None)
    return isinstance(backend, ManagedCorpusBackend) and backend.tenant == tenant_id


def begin(st, tenant_id: str, message: str, snap: Optional[Snapshot], day, extra_context: str, messages: list,
          max_answer_tokens: int, server_lookup: Optional[Callable[[], object]] = None,
          transport: Optional[httpx.BaseTransport] = None) -> Optional[Shadow]:
    """Start the shadow comparison for one exchange, or None when it does not apply (off, legacy
    tenant, no ready Rust process) or is skipped. Never raises."""
    if snap is None or mode() != "shadow" or tenant_id == "legacy":
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
        if not _managed_state_of(st, tenant_id):
            return _skip("not the tenant's managed storage")
        if getattr(st.assembler, "local_reference", ""):
            return _skip("local reference")
        if snap.embedding is None:
            return _skip("no shared embedding configuration")
        cfg = st.assembler.cfg
        request = {
            "mode": "context",
            "message": message,
            "turns": snap.turns,
            "day": day.isoformat(),
            "extra_context": extra_context,
            "trace": False,
            "max_answer_tokens": int(max_answer_tokens),
            "cfg": {"context": cfg.get("context") or {}, "retrieval": cfg.get("retrieval") or {}},
            "system_prompt": st.assembler._system_prompt(),
            "stream": False,
            "embedding": snap.embedding,
        }
        json.dumps(request)  # a config value JSON cannot carry: skipped, not failed later
        return Shadow(server.base_url, token, tenant_id, request, messages, transport=transport).start()
    except (TypeError, ValueError, OSError) as exc:
        return _skip(type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 - the shadow must never fail an exchange; no traceback (content)
        log.warning("Diary answer shadow: could not start (%s)", type(exc).__name__)
        _count("failed")
        return None


def _skip(reason: str) -> None:
    _count("skipped")
    log.info("Diary answer shadow: skipped (%s)", reason)
    return None


# ---------------- DIARY_RUST_ANSWER=on ----------------


class Declined(Exception):
    """Python answers this exchange itself (counted; nothing was sent to the model)."""


class Answered:
    """The Rust side's answer: what Python's own model call and marker split would give."""

    def __init__(self, visible: str, marker: Optional[str], truncated: bool, trace):
        self.visible = visible
        self.marker = marker
        self.truncated = truncated
        self.trace = trace


class _Trace:
    """The Rust side's retrieval trace, in the shape ``retrieval_trace.attach`` reads."""

    def __init__(self, data):
        self.data = data
        self.failed = not isinstance(data, dict)


def trace_of(answered: Answered) -> _Trace:
    """The trace to attach for a traced exchange the Rust side answered (none when it sent none)."""
    return _Trace(answered.trace)


def _python_failure(failure, setting: str) -> BaseException:
    """The exception Python's own model call raises for a failure the Rust side reports."""
    from .deadline import ExchangeTimeout
    from .llm import EmptyAnswerError, EngineUnreachable, LengthCapError, LLMError, ModelNotServed
    failure = failure if isinstance(failure, dict) else {}
    message = str(failure.get("message") or "the Rust answer failed")[:500]
    kind = failure.get("kind")
    if kind == "timeout":
        return ExchangeTimeout(message)
    if kind == "model":
        cls = failure.get("class")
        if cls == "LengthCapError":
            return LengthCapError()
        if cls == "EmptyAnswerError":
            return EmptyAnswerError()
        if cls == "ModelNotServed":
            return ModelNotServed(setting, message)
        if cls == "EngineUnreachable":
            return EngineUnreachable(message)
    return LLMError(message)


def _answered(data, setting: str) -> Answered:
    from .llm import LLMError
    if not isinstance(data, dict):
        raise LLMError("unreadable Rust answer")
    if "failed" in data:
        raise _python_failure(data.get("failed"), setting)
    visible, marker = data.get("visible"), data.get("marker")
    if not isinstance(visible, str) or marker not in (None, "ok", "skip"):
        raise LLMError("unreadable Rust answer")
    return Answered(visible, marker, data.get("truncated") is True, data.get("trace"))


def answer(st, tenant_id: str, message: str, snap: Optional[Snapshot], day, extra_context: str, max_answer_tokens: int,
           budget, emit=None, trace: bool = False, server_lookup: Optional[Callable[[], object]] = None,
           transport: Optional[httpx.BaseTransport] = None) -> Answered:
    """The companion's answer from the Rust process, within ``budget``. Raises Declined when Python
    must answer itself; otherwise the answer, or the exception Python's model call would raise.
    ``emit`` (a streamed exchange) receives the model's events as the Rust side relays them."""
    if snap is None or not answering() or tenant_id == "legacy":
        raise Declined("off")
    try:
        if server_lookup is None:
            from . import rust_forward
            server_lookup = rust_forward.current
        server = server_lookup()
        token, ready = server.control() if server is not None else ("", False)
        if not ready or not token:
            raise _decline("unavailable", "no ready Rust process")
        if not _managed_state_of(st, tenant_id):
            raise _decline("skipped", "not the tenant's managed storage")
        if getattr(st.assembler, "local_reference", ""):
            raise _decline("skipped", "local reference")
        if snap.embedding is None:
            raise _decline("skipped", "no shared embedding configuration")
        cfg = st.assembler.cfg
        request = {
            "mode": "answer",
            "message": message,
            "turns": snap.turns,
            "day": day.isoformat(),
            "extra_context": extra_context,
            "trace": bool(trace),
            "max_answer_tokens": int(max_answer_tokens),
            "cfg": {"context": cfg.get("context") or {}, "retrieval": cfg.get("retrieval") or {}},
            "system_prompt": st.assembler._system_prompt(),
            "stream": emit is not None,
            "embedding": snap.embedding,
        }
        body = json.dumps(request)
    except Declined:
        raise
    except Exception as exc:  # noqa: BLE001 - Python answers; no content in the log
        raise _decline("skipped", type(exc).__name__)
    setting = str(getattr(getattr(st, "llm_main", None), "setting_key", "") or "")
    started = threading.Event()  # set once the Rust side has accepted (200): no fallback after it

    def call(gate, alive):
        from .rust_forward import CONTROL_HEADER, DECLINE_HEADER
        from .llm import LLMError, _shutdown_upstream
        holder = {}

        def trace(name, info):
            # The connection as soon as it exists: a reply whose headers have not come yet (the plain
            # answer replies only when the model has finished) can still be shut when it is given up.
            if name == "connection.connect_tcp.complete" and isinstance(info, dict):
                holder["network"] = info.get("return_value")
                if getattr(alive, "abandoned", False):
                    _shutdown_upstream(holder)  # given up while connecting: the closer found nothing
        on_abandon = getattr(alive, "on_abandon", None)
        unregister = on_abandon(lambda: _shutdown_upstream(holder)) if callable(on_abandon) else None
        try:
            with httpx.Client(base_url=server.base_url, timeout=ANSWER_TIMEOUT, trust_env=False,
                              transport=transport) as client:
                with client.stream("POST", "/internal/answer", params={"tenant": tenant_id}, content=body,
                                   headers={CONTROL_HEADER: token, "content-type": "application/json"},
                                   extensions={"trace": trace}) as resp:
                    holder["response"] = resp
                    if resp.status_code == 421 and resp.headers.get(DECLINE_HEADER.decode()) == "1":
                        raise _decline("declined", "")
                    if resp.status_code >= 500:
                        # The Rust side may have called the model before failing: never a second answer.
                        raise LLMError(f"Rust answer failed: HTTP {resp.status_code}")
                    if resp.status_code != 200:
                        raise _decline("unavailable", f"HTTP {resp.status_code}")
                    started.set()
                    if emit is None:
                        return _answered(json.loads(resp.read()), setting)
                    for line in resp.iter_lines():
                        alive()
                        if not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:].strip())
                        kind = event.get("type") if isinstance(event, dict) else None
                        if kind == "answered":
                            return _answered(event, setting)
                        if kind == "failed":
                            raise _python_failure(event.get("failure"), setting)
                        if kind is not None:
                            gate(event)
                    raise LLMError("the Rust answer stream ended before its outcome")
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if started.is_set():
                raise LLMError(f"Rust answer failed: {type(exc).__name__}") from exc
            raise _decline("unavailable", type(exc).__name__) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Rust answer failed: {type(exc).__name__}") from exc
        finally:
            if unregister is not None:
                unregister()

    try:
        result = budget.run(call, emit=emit, first_token=False)
    except Declined:
        raise
    except BaseException:
        _count_answer("rust")  # the Rust side took the exchange: its failure is the exchange's
        raise
    _count_answer("rust")
    return result


def _decline(outcome: str, reason: str) -> Declined:
    _count_answer(outcome)
    if reason:
        log.info("Diary answer: Python answers (%s: %s)", outcome, reason)
    else:
        log.info("Diary answer: Python answers (%s)", outcome)
    return Declined(outcome)
