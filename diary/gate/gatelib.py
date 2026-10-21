"""Shared helpers for the Diary gate: env file, signed tenant client, evidence layout, load gate.

Standard library only, so every gate script runs with `python3 -I` on the host and needs no venv.
Secrets (DIARY_AUTH_TOKEN, DIARY_TENANT_KEY, LLM_API_KEY) are read from .env.gate and are never
printed, logged or written to evidence.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

GATE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = GATE_DIR / ".env.gate"
# The gate diary (inference per .env.gate) and the no-model diary (inference always disabled) of compose.yml.
DEFAULT_BASE_URL = os.environ.get("GATE_BASE_URL") or f"http://127.0.0.1:{os.environ.get('GATE_PORT', '18910')}"
NOMODEL_BASE_URL = (os.environ.get("GATE_NOMODEL_BASE_URL")
                    or f"http://127.0.0.1:{os.environ.get('GATE_NOMODEL_PORT', '18911')}")
TENANT_NS = uuid.UUID("0c7b4f8e-2a55-4d3f-a1e6-9d0b6c2f4e83")
# The diary's tenant assertion (agent/tenant_assertion.py, apps/web/server/diary-tenant-assertion.cjs).
ASSERTION_HEADER = "X-Cowork-Tenant-Assertion"
ASSERTION_LABEL = "cowork-diary-tenant-v2"
SECRET_KEYS = ("DIARY_AUTH_TOKEN", "DIARY_TENANT_KEY", "LLM_API_KEY", "LLM_AUX_API_KEY")
# base64url {"kind":"local"}: lets /api/health report tenant detail (journal, retrieval) for an app-managed tenant.
LOCAL_STORAGE = "eyJraW5kIjoibG9jYWwifQ"
DISABLED_INFERENCE = "http://gate-inference-disabled.invalid:9/v1"
STATUSES = ("pass", "fail", "blocked", "untested")


def load_env(path=None) -> dict:
    """KEY=VALUE lines; blank lines and # comments ignored; no shell expansion."""
    env = {}
    p = Path(path or DEFAULT_ENV)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def inference_enabled(env: dict) -> bool:
    url = env.get("LLM_BASE_URL", "")
    return bool(url) and "inference-disabled" not in url


def tenant_id(label: str) -> str:
    """A fixed synthetic tenant per label (each area has its own; the stack is fresh per run)."""
    return str(uuid.uuid5(TENANT_NS, f"diarygate:{label}"))


def _sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()


def sign(key: str, user_id: str, method: str, path: str, query: bytes = b"", body: bytes = b"",
         body_hash: str | None = None, storage: str = "", legacy_owner: str = "", blocked: str = "",
         ts: int | None = None, nonce: str | None = None) -> str:
    """v2.<ts>.<nonce>.<b64url HMAC-SHA256> over the canonical string the diary verifies."""
    ts = int(time.time()) if ts is None else ts
    nonce = secrets.token_hex(16) if nonce is None else nonce
    message = "\n".join([ASSERTION_LABEL, user_id.lower(), str(ts), nonce, method.upper(), path,
                         "query=" + _sha(query), "body=" + (body_hash if body_hash is not None else _sha(body)),
                         _sha(storage), legacy_owner, blocked])
    sig = base64.urlsafe_b64encode(hmac.new(key.encode(), message.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    return f"v2.{ts}.{nonce}.{sig}"


class Response:
    def __init__(self, status: int, body: bytes, headers: dict):
        self.status, self.body, self.headers = status, body, headers

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None


GATE_HOSTS = ("127.0.0.1", "localhost")


# compose.yml publishes the two gate diaries in this band only (defaults 18910 and 18911).
GATE_PORT_RANGE = range(18910, 18920)
# Ports production services publish or listen on (MS-01 inference compose; noevia compose.yaml and
# deploy overrides): never a gate port, whatever the environment says.
PRODUCTION_PORTS = frozenset({8010, 8021, 8031, 8040, 8080, 8082, 8092, 11435, 18040, 18080})


def gate_ports() -> tuple:
    """(gate diary port, no-model diary port). GATE_PORT / GATE_NOMODEL_PORT may move them within
    GATE_PORT_RANGE only; they must differ and never be a production port."""
    try:
        gate = int(os.environ.get("GATE_PORT", "18910"))
        nomodel = int(os.environ.get("GATE_NOMODEL_PORT", "18911"))
    except ValueError:
        raise SystemExit("refused: GATE_PORT and GATE_NOMODEL_PORT must be integers")
    for name, port in (("GATE_PORT", gate), ("GATE_NOMODEL_PORT", nomodel)):
        if port not in GATE_PORT_RANGE or port in PRODUCTION_PORTS:
            raise SystemExit(f"refused: {name}={port} is outside the gate port range "
                             f"{GATE_PORT_RANGE.start}-{GATE_PORT_RANGE.stop - 1} or a production port")
    if gate == nomodel:
        raise SystemExit("refused: GATE_PORT and GATE_NOMODEL_PORT must differ")
    return gate, nomodel


def check_base_url(url: str, nomodel: bool = False) -> str:
    """Hard isolation guard, applied before any request: plain http on 127.0.0.1 or localhost, on one of
    the two gate ports; a no-model client may use only the no-model port. Anything else is refused."""
    try:
        parsed = urllib.parse.urlsplit(str(url))
        port = parsed.port
    except ValueError:
        raise SystemExit(f"refused base URL {url!r}: not a URL")
    gate, nomodel_port = gate_ports()
    allowed = (nomodel_port,) if nomodel else (gate, nomodel_port)
    bad_target = parsed.scheme != "http" or parsed.hostname not in GATE_HOSTS or port not in allowed
    extras = parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment
    if bad_target or extras:
        raise SystemExit(f"refused base URL {url!r}: only http://127.0.0.1 or localhost on port "
                         f"{' or '.join(map(str, allowed))} ({'no-model diary' if nomodel else 'gate diaries'})")
    return f"http://{parsed.hostname}:{port}"


class RedirectRefused(urllib.error.HTTPError):
    """A gate request was answered with a redirect; it is never followed (the bearer token would go along)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RedirectRefused(req.full_url, code, f"redirect refused ({code})", headers, fp)


# Every gate request (diary API and load probes) goes through this opener: no proxy from the
# environment (http_proxy / no_proxy are ignored) and no redirects.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


class Client:
    """Talks to a gate diary as the web server would: bearer token plus a signed tenant assertion.
    The base URL is checked by check_base_url before any request (nomodel=True: the no-model diary only)."""

    def __init__(self, base_url: str, env: dict, timeout: float = 30.0, log=None, nomodel: bool = False):
        self.base_url = check_base_url(base_url, nomodel)
        self.token = env.get("DIARY_AUTH_TOKEN", "")
        self.key = env.get("DIARY_TENANT_KEY", "")
        self.timeout = timeout
        self.log = log  # callable(dict) for evidence: method, path, status, ms (never headers or bodies)

    def request(self, method: str, path: str, user: str | None = None, query: str = "", json_body=None,
                raw_body: bytes | None = None, content_type: str | None = None, sign_as: str | None = None,
                assertion: str | None = None, bearer: bool = True, extra_headers: dict | None = None,
                timeout: float | None = None, storage: str = "") -> Response:
        body = raw_body if raw_body is not None else (b"" if json_body is None else
                                                      json.dumps(json_body, ensure_ascii=False).encode("utf-8"))
        headers = {"Accept": "application/json"}
        if json_body is not None or content_type:
            headers["Content-Type"] = content_type or "application/json"
        if bearer and self.token:
            headers["Authorization"] = "Bearer " + self.token
        if user:
            headers["X-Cowork-User-ID"] = user
            if storage:
                headers["X-Cowork-Storage"] = storage
            if assertion == "omit":
                pass
            elif assertion is not None:
                headers[ASSERTION_HEADER] = assertion
            elif self.key:
                hashed = (content_type or "application/json").split(";")[0].strip().lower() in ("", "application/json")
                headers[ASSERTION_HEADER] = sign(self.key, sign_as or user, method, urllib.parse.quote(path, safe="/%-._~"),
                                                 query.encode(), body, None if hashed else "stream", storage=storage)
        headers.update(extra_headers or {})
        url = self.base_url + urllib.parse.quote(path, safe="/%-._~") + ("?" + query if query else "")
        # No data for an empty body: urllib would add a form Content-Type, which the diary signs as "stream".
        req = urllib.request.Request(url, data=body or None,
                                     method=method, headers=headers)
        started = time.monotonic()
        try:
            with OPENER.open(req, timeout=timeout or self.timeout) as resp:
                result = Response(resp.status, resp.read(), dict(resp.headers))
        except urllib.error.HTTPError as exc:
            result = Response(exc.code, exc.read(), dict(exc.headers or {}))
        except (urllib.error.URLError, OSError) as exc:
            # Transport failure (refused, reset mid-upload, timeout): status 0, never an exception.
            result = Response(0, json.dumps({"transport_error": type(exc).__name__}).encode(), {})
        if self.log:
            self.log({"method": method, "path": path, "query": query, "status": result.status,
                      "ms": round((time.monotonic() - started) * 1000)})
        return result



# ---------------- evidence layout ----------------

def _inside_git(path: Path) -> bool:
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return True
    return False


def evidence_dir(root, rev: str, phase: str) -> Path:
    """<root>/release-evidence/diary-gate/<rev>-<pre|post>/ with the fixed layout; refused inside a git tree."""
    if phase not in ("pre", "post"):
        raise SystemExit("--phase must be pre or post")
    root = Path(root).expanduser().resolve()
    if _inside_git(root):
        raise SystemExit(f"evidence root {root} is inside a git work tree; pass a directory outside git")
    run = root / "release-evidence" / "diary-gate" / f"{rev[:12]}-{phase}"
    for sub in ("stages", "workflows", "logs"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    blockers = run / "blockers.md"
    if not blockers.exists():
        blockers.write_text("# Blockers\n\n", encoding="utf-8")
    return run


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def add_blocker(run: Path, text: str) -> None:
    with open(run / "blockers.md", "a", encoding="utf-8") as fh:
        fh.write(f"- {text}\n")


# ---------------- MS-01 load gate (before every inference call and every embedding write) ----------------

MIN_MEM_AVAILABLE = 6 * 1024 ** 3
# A private directory (mode 0700) under the gate work dir, never a predictable shared /tmp path.
LOCK_DIR = os.path.expanduser(os.environ.get("GATE_LOCK_DIR", "~/noevia-work/diarygate-work/locks"))
# Bounded waits. Each reports "blocked" with its reason when it runs out; none turns into a query failure.
#   GATE_LOCK_TIMEOUT_S   wait for the cross-process inference lock held by another gate process
#   GATE_RECHECKS/_S      load-gate re-checks (a previous turn's background reindex may still embed)
#   GATE_IDLE_WAIT_S/_POLL_S  wait for the engines to go idle before a restore and before the next query
#                         (a chat call the diary gave up on may still be generating upstream)
#   GATE_AREA_WAIT_S      wait for another model area to finish (model areas run one at a time)
#   GATE_MODEL_AREAS_PARALLEL=1  opt in to running model areas at the same time
LOCK_TIMEOUT_S = float(os.environ.get("GATE_LOCK_TIMEOUT_S", "1800"))
RECHECKS = int(os.environ.get("GATE_RECHECKS", "12"))
RECHECK_S = float(os.environ.get("GATE_RECHECK_S", "5"))
IDLE_WAIT_S = float(os.environ.get("GATE_IDLE_WAIT_S", "900"))
IDLE_POLL_S = float(os.environ.get("GATE_IDLE_POLL_S", "5"))
AREA_WAIT_S = float(os.environ.get("GATE_AREA_WAIT_S", "3600"))
# The longest settle() waits for the journal and the index (restore_baseline); run_areas.py sizes its stop grace on it.
SETTLE_TIMEOUT_S = 300.0
# The longest the gate waits for one chat call (harness.ask). run_areas.py derives its default stop grace from it.
CHAT_TIMEOUT_S = 600.0


def model_areas_parallel() -> bool:
    """Model areas run serially unless GATE_MODEL_AREAS_PARALLEL is 1/true/yes."""
    return os.environ.get("GATE_MODEL_AREAS_PARALLEL", "").strip().lower() in ("1", "true", "yes")


_STOP = {"requested": False}


def stop_requested() -> bool:
    """True once SIGTERM arrived after install_stop_handler (run_areas.py sends it when an area overruns)."""
    return _STOP["requested"]


STOPPED_REASON = "stopped by SIGTERM"


def stopped_block() -> "GateBlocked":
    """The blocked result for a wait that a stop request cut short (kind "stopped")."""
    return GateBlocked({"ok": False, "kind": "stopped", "reasons": [STOPPED_REASON]})


def install_stop_handler() -> None:
    """Make SIGTERM a request to stop at the next safe point instead of an immediate death: the running
    query finishes and its tenant is restored to the baseline, then the area stops as blocked. Only the
    entry points that restore a tenant (model areas, harness queries) install it."""
    def handler(signum, frame):  # noqa: ARG001
        _STOP["requested"] = True
    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:  # not the main thread
        pass


class GateBlocked(Exception):
    """The load gate refused (after its bounded re-checks). result holds the reasons."""

    def __init__(self, result: dict):
        result.setdefault("status", "blocked")  # a refusal is "blocked" with a reason, never a failed query
        self.result = result
        super().__init__("; ".join(result.get("reasons") or ["load gate refused"]))


def _get_json(url: str, token: str = "", timeout: float = 5.0):
    headers = {"Authorization": "Bearer " + token} if token else {}
    with OPENER.open(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def engine_models(base: str, token: str = "", fetch=_get_json) -> tuple:
    """([(model id, state)], routed) from the engine's model list. This is the only thing the gate may ask
    of an engine that might be asleep: listing models never wakes one. state is 'loaded', 'unloaded',
    'loading' or whatever the router reports. Only a list in which NO entry has a status key means an engine
    without a router (routed False; a single engine answers plain /slots). As soon as any entry carries a
    status the list is routed, and an entry with a missing or malformed status is 'unknown' (busy, never
    probed). Raises when unreadable."""
    data = fetch(base.rstrip("/") + "/v1/models", token).get("data") or []
    routed = any(isinstance(m, dict) and "status" in m for m in data)
    models = []
    for m in data:
        if not isinstance(m, dict):
            models.append((None, "unknown"))
            continue
        status = m.get("status")
        value = status.get("value") if isinstance(status, dict) else (status if isinstance(status, str) else None)
        if routed:
            models.append((m.get("id"), value if isinstance(value, str) and value else "unknown"))
        else:
            models.append((m.get("id"), "loaded"))
    return models, routed


def _count_processing(slots: list) -> int:
    return sum(1 for s in slots if not isinstance(s, dict) or s.get("is_processing") is not False)


def _slots_busy(base: str, token: str = "", fetch=_get_json) -> tuple:
    """(busy, detail) for one engine. EVERY load-gate probe goes through here: it reads the model list
    first and requests /slots ONLY for a model the list says is loaded, because a slots request sent to a
    sleeping model on the router wakes it. An unloaded model is idle by definition (nothing to wait for);
    a loading model counts as busy; anything unreadable or unrecognised fails closed."""
    base = base.rstrip("/")
    try:
        models, routed = engine_models(base, token, fetch)
    except Exception as exc:  # noqa: BLE001
        return True, f"load unknown ({type(exc).__name__}, model list unreadable); refusing to start"
    if not models:
        return True, "load unknown (model list empty); refusing to start"
    loaded = [mid for mid, state in models if state == "loaded"]
    unloaded = sum(1 for _, state in models if state == "unloaded")
    loading = sum(1 for _, state in models if state == "loading")
    other = sorted({str(state) for _, state in models if state not in ("loaded", "unloaded", "loading")})
    notes, busy = [], False
    if loading:
        busy = True
        notes.append(f"{loading} model(s) loading")
    if other:
        busy = True
        notes.append(f"model state not recognised ({', '.join(other)})")
    if unloaded:
        notes.append(f"{unloaded} model(s) unloaded, slots not probed")
    if loaded:
        processing, seen, unread = 0, 0, 0
        if not routed:
            # No status anywhere: a single engine, which answers plain /slots itself. Never fall back to
            # per-model probes here: on a router they would wake every listed model. More than one model, or
            # a plain /slots that fails, means we cannot tell which models are loaded: fail closed.
            plain = None
            if len(models) == 1:
                try:
                    plain = fetch(base + "/slots", token)
                except Exception:  # noqa: BLE001
                    plain = None
            if not isinstance(plain, list):
                return True, "; ".join(notes + ["cannot tell which models are loaded (model list has no status); refusing to start"])
            processing, seen = _count_processing(plain), 1
        else:
            for mid in loaded:  # loaded per the model list, so this cannot wake anything
                try:
                    slots = fetch(base + "/slots?model=" + urllib.parse.quote(str(mid)), token)
                except Exception:  # noqa: BLE001
                    unread += 1
                    continue
                if not isinstance(slots, list):
                    unread += 1
                    continue
                seen += 1
                processing += _count_processing(slots)
        if unread or not seen:
            busy = True
            notes.append(f"slots unreadable for {unread} loaded model(s), readable for {seen}; refusing to start")
        else:
            busy = busy or processing > 0
            notes.append(f"{processing} slot(s) processing across {seen} model(s)")
    return busy, "; ".join(notes)


def mem_available(path: str = "/proc/meminfo"):
    """Bytes of MemAvailable, or None when it cannot be read (the gate then blocks)."""
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _strip_v1(url: str) -> str:
    url = (url or "").rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def load_targets(env: dict) -> list:
    """(role, url, token) for every engine the diary calls: embed, chat (kev) and aux (unless it is the chat engine)."""
    key = env.get("LLM_API_KEY", "")
    kev = env.get("GATE_KEV_LOAD_URL", "")
    aux = env.get("GATE_AUX_LOAD_URL") or _strip_v1(env.get("LLM_AUX_BASE_URL", ""))
    targets = [("embed", env.get("GATE_EMBED_LOAD_URL", ""), key), ("kev", kev, key)]
    if not aux or aux.rstrip("/") != kev.rstrip("/"):
        targets.append(("aux", aux, env.get("LLM_AUX_API_KEY") or key))
    return targets


def engine_load(env: dict, fetch=_get_json) -> tuple:
    """(ok, reasons, details): every engine the diary calls is idle and readable. Fails closed."""
    ok, reasons, details = True, [], {}
    for role, url, token in load_targets(env):
        if not url:
            ok = False
            reasons.append(f"{role}: no load URL; cannot check its load")
            continue
        busy, detail = _slots_busy(url, token, fetch)
        details[role] = detail
        if busy:
            ok = False
            reasons.append(f"{role}: {detail}")
    return ok, reasons, details


def load_gate(env: dict, fetch=_get_json, meminfo: str = "/proc/meminfo") -> dict:
    """ok=False when MemAvailable is under 6 GiB or unreadable, or any engine has a request in flight or
    cannot be read. Fails closed."""
    result = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ok": True, "reasons": []}
    mem = mem_available(meminfo)
    if mem is None:
        result["ok"] = False
        result["reasons"].append("MemAvailable unreadable")
    else:
        result["mem_available_gib"] = round(mem / 1024 ** 3, 2)
        if mem < MIN_MEM_AVAILABLE:
            result["ok"] = False
            result["reasons"].append(f"MemAvailable {result['mem_available_gib']} GiB < 6 GiB")
    ok, reasons, details = engine_load(env, fetch)
    result.update(details)
    result["ok"] = result["ok"] and ok
    result["reasons"].extend(reasons)
    return result


def require_gate(env: dict, rechecks: int = RECHECKS, wait: float = RECHECK_S, check=load_gate, sleep=time.sleep) -> dict:
    """Re-check a bounded number of times (a previous turn's background reindex may still be embedding);
    raise GateBlocked when it never clears."""
    result = {}
    for attempt in range(1, max(1, rechecks) + 1):
        result = check(env)
        result["attempt"] = attempt
        if result["ok"]:
            return result
        if attempt < rechecks:
            sleep(wait)
    result.setdefault("kind", "load-gate")
    raise GateBlocked(result)


def wait_for_idle(env: dict, timeout: float | None = None, poll: float | None = None, fetch=None,
                  sleep=None, clock=None, stoppable: bool = True) -> dict:
    """Poll the engines (model list first, slots only for loaded models) until none has a request in
    flight, for at most `timeout` seconds (GATE_IDLE_WAIT_S). A chat call the diary gave up on can keep
    generating upstream, so the next restore write or query would be refused while it runs. Returns
    {"idle", "waited_s", "polls", "reasons"}; never raises. Does nothing when inference is disabled.
    With `stoppable` (the default) a stop request (SIGTERM after install_stop_handler) ends the wait at the
    next poll: idle is False, "stopped" is True and the reason is "stopped by SIGTERM". The wait before a
    restore passes stoppable=False: the restore is what a stop request is waiting for."""
    if not inference_enabled(env):
        return {"idle": True, "waited_s": 0.0, "polls": 0, "reasons": [], "skipped": "inference disabled"}
    fetch, sleep, clock = fetch or _get_json, sleep or time.sleep, clock or time.monotonic
    timeout = IDLE_WAIT_S if timeout is None else timeout
    poll = max(IDLE_POLL_S if poll is None else poll, 0.1)
    start, polls = clock(), 0
    while True:
        polls += 1
        if stoppable and stop_requested():
            return {"idle": False, "stopped": True, "waited_s": round(clock() - start, 3), "polls": polls,
                    "reasons": [STOPPED_REASON]}
        ok, reasons, _ = engine_load(env, fetch)
        waited = clock() - start
        if ok:
            return {"idle": True, "waited_s": round(waited, 3), "polls": polls, "reasons": []}
        if waited >= timeout:
            return {"idle": False, "waited_s": round(waited, 3), "polls": polls, "reasons": reasons}
        sleep(min(poll, timeout - waited))


def require_idle(env: dict, log=None, before: str = "", **kw) -> dict:
    """wait_for_idle, recorded in the evidence (`log` gets {"kind": "idle-wait", "before", "waited_s", ...}),
    and a GateBlocked (status "blocked", kind "not-idle") when the engines never go idle, or kind "stopped"
    ("stopped by SIGTERM") when a stop request ended the wait (stoppable=True, the default)."""
    res = wait_for_idle(env, **kw)
    if res.get("skipped"):
        return res
    record = {"kind": "idle-wait", "before": before, "idle": res["idle"], "waited_s": res["waited_s"], "polls": res["polls"]}
    if res.get("stopped"):
        record["stopped"] = True
    if not res["idle"]:
        record["reasons"] = res["reasons"]
    if log:
        log(record)
    if res.get("stopped"):
        raise stopped_block()
    if not res["idle"]:
        limit = IDLE_WAIT_S if kw.get("timeout") is None else kw["timeout"]
        raise GateBlocked({"ok": False, "kind": "not-idle", "idle_wait_s": res["waited_s"],
                           "reasons": [f"engines not idle after {res['waited_s']:.0f} s (limit {limit:.0f} s, "
                                       f"GATE_IDLE_WAIT_S): " + "; ".join(res["reasons"])]})
    return res


def _lock_path(lock_file: str | None, name: str = "inference.lock") -> str:
    if lock_file:
        return lock_file
    os.makedirs(LOCK_DIR, mode=0o700, exist_ok=True)
    os.chmod(LOCK_DIR, 0o700)
    return os.path.join(LOCK_DIR, name)


def _holder(path: str) -> str:
    """Who last took the lock (pid and script), for the blocked reason; empty when unknown."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()[:120]
    except OSError:
        return ""


def _acquire(fh, path: str, timeout: float, sleep=time.sleep, clock=time.monotonic, what: str = "inference lock",
             kind: str = "lock-timeout", env_name: str = "GATE_LOCK_TIMEOUT_S", stoppable: bool = True) -> float:
    """flock with a bounded wait; returns the seconds waited. Fails closed with GateBlocked (status
    "blocked", with the holder when known) when another gate process holds it too long. While the lock is
    held by someone else, a stop request ends the wait at the next poll with kind "stopped"
    ("stopped by SIGTERM"); stoppable=False (the restore path) keeps waiting for the bound."""
    start = clock()
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.seek(0)
            fh.truncate(0)
            fh.write(f"pid={os.getpid()} {os.path.basename(sys.argv[0] or 'gate')}\n")
            fh.flush()
            return clock() - start
        except BlockingIOError:
            waited = clock() - start
            if stoppable and stop_requested():
                exc = stopped_block()
                exc.result["lock_wait_s"] = round(waited, 3)
                raise exc
            if waited >= timeout:
                holder = _holder(path)
                raise GateBlocked({"ok": False, "kind": kind, "lock_wait_s": round(waited, 3),
                                   "reasons": [f"{what} still held after {waited:.0f} s (limit {timeout:.0f} s, {env_name})"
                                               + (f" by {holder}" if holder else "")]})
            sleep(0.5)


@contextmanager
def inference_slot(env: dict, log=None, lock_file: str | None = None, lock_timeout: float | None = None,
                   stoppable: bool = True):
    """Hold a cross-process lock around the load check AND the call it guards, so no two gate processes
    start inference work at once. A no-op when inference is disabled (nothing can reach a model).
    The lock wait is bounded (GATE_LOCK_TIMEOUT_S): when it runs out the slot raises GateBlocked (status
    "blocked", kind "lock-timeout", who holds it) before the guarded call is made, so callers report the
    work as blocked, not as a query that timed out. A stop request ends the lock wait early (kind "stopped")
    unless stoppable=False, which the restore path uses: a restore must finish after a stop request."""
    if not inference_enabled(env):
        yield None
        return
    if stoppable and stop_requested():  # no new inference work once a stop was asked for (the fixture load, a query)
        exc = stopped_block()
        if log:
            log(exc.result)
        raise exc
    path = _lock_path(lock_file)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+") as fh:
        try:
            lock_wait = _acquire(fh, path, LOCK_TIMEOUT_S if lock_timeout is None else lock_timeout, stoppable=stoppable)
        except GateBlocked as exc:
            if log:
                log(exc.result)
            raise
        try:
            try:
                result = require_gate(env)
            except GateBlocked as exc:
                exc.result["lock_wait_s"] = round(lock_wait, 3)
                if log:
                    log(exc.result)
                raise
            result["lock_wait_s"] = round(lock_wait, 3)
            if log:
                log(result)
            yield result
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


@contextmanager
def model_area_slot(name: str, log=None, lock_file: str | None = None, timeout: float | None = None):
    """Model areas run one at a time: four of them sharing the engines starve each other on the inference
    lock and their queries time out. The area holds this lock for its whole run. Waiting is bounded
    (GATE_AREA_WAIT_S); when it runs out GateBlocked (status "blocked", kind "area-wait-timeout") names the
    holder. GATE_MODEL_AREAS_PARALLEL=1 skips the lock."""
    if model_areas_parallel():
        yield {"serial": False, "waited_s": 0.0}
        return
    path = _lock_path(lock_file, "model-area.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+") as fh:
        try:
            waited = _acquire(fh, path, AREA_WAIT_S if timeout is None else timeout, what=f"model-area lock (area {name} waiting)",
                              kind="area-wait-timeout", env_name="GATE_AREA_WAIT_S")
        except GateBlocked as exc:
            if log:
                log(exc.result)
            raise
        try:
            yield {"serial": True, "waited_s": round(waited, 3)}
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ---------------- fixture through the API ----------------

def tree_hash(files: dict) -> str:
    """Same algorithm as gen_fixture.tree_hash."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\0{files[rel]}\n".encode())
    return h.hexdigest()


def verify_fixture(fixture_dir) -> dict:
    """Recompute every manifest hash (fixture files and truth/*.jsonl) and the tree hash. Any missing,
    changed or unlisted file is a clean refusal (SystemExit), never a KeyError later."""
    root = Path(fixture_dir)
    mpath = root / "manifest.json"
    if not mpath.is_file():
        raise SystemExit(f"fixture refused: {mpath} is missing")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        listed = dict(manifest["files"])
        expected_tree = manifest["tree_sha256"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"fixture refused: manifest unreadable ({type(exc).__name__})")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != mpath}
    unlisted, missing = sorted(actual - set(listed)), sorted(set(listed) - actual)
    if unlisted:
        raise SystemExit(f"fixture refused: {len(unlisted)} file(s) not in the manifest, e.g. {unlisted[0]}")
    if missing:
        raise SystemExit(f"fixture refused: {len(missing)} manifest file(s) missing, e.g. {missing[0]}")
    changed = sorted(rel for rel in listed if _sha((root / rel).read_bytes()) != listed[rel])
    if changed:
        raise SystemExit(f"fixture refused: {len(changed)} file(s) differ from the manifest, e.g. {changed[0]}")
    if tree_hash(listed) != expected_tree:
        raise SystemExit("fixture refused: tree hash does not match the manifest")
    return manifest


def fixture_files(fixture_dir: Path) -> list:
    """Corpus-relative paths of every fixture day file, in sorted order."""
    root = Path(fixture_dir) / "fixture"
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.md"))


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_fixture(client: Client, user: str, fixture_dir: Path, env: dict | None = None, gate_log=None) -> dict:
    """PUT every day file through the workspace file API (version null = must not exist yet). Each PUT
    reindexes (embeds) synchronously, so each is gated when inference is enabled."""
    root, failures, written = Path(fixture_dir) / "fixture", [], 0
    for rel in fixture_files(fixture_dir):
        content = (root / rel).read_text(encoding="utf-8")
        with inference_slot(env or {}, gate_log):
            r = client.request("PUT", "/api/file", user=user, json_body={"path": rel, "content": content, "version": None},
                               timeout=300)
        if r.status == 200 and (r.json() or {}).get("version") == _sha(content.encode()):
            written += 1
        else:
            failures.append({"path": rel, "status": r.status})
    return {"written": written, "failures": failures}


def readback(client: Client, user: str, fixture_dir: Path, mutated: bool = False, env: dict | None = None, gate_log=None) -> dict:
    """Read every day back: file API byte-exact against the manifest (before mutations), month API with
    every xid marker present (after mutations the correction entries too). One gated batch: the first
    read of a tenant can run its recovery reindex."""
    manifest = json.loads((Path(fixture_dir) / "manifest.json").read_text(encoding="utf-8"))
    entries = read_jsonl(Path(fixture_dir) / "truth" / "entries.jsonl")
    mismatched, months, missing = [], {}, []
    with inference_slot(env or {}, gate_log):
        for rel in fixture_files(fixture_dir):
            r = client.request("POST", "/api/file", user=user, json_body={"path": rel}, timeout=300)
            content = (r.json() or {}).get("content") if r.status == 200 else None
            want = manifest["files"].get("fixture/" + rel)
            if content is None or (not mutated and _sha(content.encode()) != want):
                mismatched.append({"path": rel, "status": r.status})
        for e in entries:
            if e["id"].startswith("C") and not mutated:
                continue
            month = e["day"][:7]
            if month not in months:
                r = client.request("GET", "/api/day", user=user, query="month=" + month)
                months[month] = (r.json() or {}).get("log", "") if r.status == 200 else ""
            if f"xid:{e['xid']}" not in months[month]:
                missing.append(e["id"])
    return {"files_checked": len(fixture_files(fixture_dir)), "mismatched": mismatched, "months": sorted(months),
            "markers_missing": missing}


def apply_mutations(client: Client, user: str, fixture_dir: Path, env: dict | None = None, gate_log=None) -> dict:
    """Scripted paraphrases and deletions (guarded edit by xid) and corrections (a new entry appended to a
    later day). Each write reindexes, so each is gated when inference is enabled."""
    results = []
    for m in read_jsonl(Path(fixture_dir) / "truth" / "mutations.jsonl"):
        with inference_slot(env or {}, gate_log):
            if m["op"] == "edit":
                r = client.request("POST", "/api/entries/edit", user=user,
                                   json_body={"xid": m["xid"], "me": m["me"], "assistant": m["assistant"], "month": m["month"]},
                                   timeout=300)
                results.append({"id": m["id"], "status": r.status, "ok": r.status == 200})
                continue
            cur = client.request("POST", "/api/file", user=user, json_body={"path": m["path"]})
            body = cur.json() or {}
            if cur.status != 200 or body.get("content") is None:
                results.append({"id": m["id"], "status": cur.status, "ok": False})
                continue
            new = body["content"].rstrip("\n") + "\n\n" + m["block"] + "\n"
            r = client.request("PUT", "/api/file", user=user,
                               json_body={"path": m["path"], "content": new, "version": body.get("version")}, timeout=300)
            results.append({"id": m["id"], "status": r.status, "ok": r.status == 200})
    return {"applied": sum(1 for x in results if x["ok"]), "total": len(results), "results": results}


# ---------------- baseline and restore (no query may see an earlier query's logged exchange) ----------------

def list_tree(client: Client, user: str, path: str = "") -> list:
    """Every Markdown file path under `path` (recursive, through GET /api/files)."""
    r = client.request("GET", "/api/files", user=user, query=("path=" + urllib.parse.quote(path)) if path else "")
    if r.status != 200:
        raise RuntimeError(f"listing {path or '/'} failed: http {r.status}")
    out = []
    for item in (r.json() or {}).get("files") or []:
        if item.get("isDir"):
            out.extend(list_tree(client, user, item["path"]))
        else:
            out.append(item["path"])
    return sorted(out)


def snapshot(client: Client, user: str) -> dict:
    """path -> content for every file of the tenant (the post-mutation baseline)."""
    files = {}
    for path in list_tree(client, user):
        r = client.request("POST", "/api/file", user=user, json_body={"path": path})
        if r.status != 200 or (r.json() or {}).get("content") is None:
            raise RuntimeError(f"snapshot read of {path} failed: http {r.status}")
        files[path] = r.json()["content"]
    return files


def health(client: Client, user: str) -> dict:
    r = client.request("GET", "/api/health", user=user, storage=LOCAL_STORAGE)
    return (r.json() or {}) if r.status == 200 else {}


def settle(client: Client, user: str, expect_chunks=None, timeout: float = SETTLE_TIMEOUT_S, sleep=time.sleep) -> dict:
    """Wait until the journal is empty and (with inference) the index holds the baseline chunk count,
    twice in a row. Returns {"ok": bool, ...}; never raises."""
    deadline, stable, last = time.monotonic() + timeout, 0, {}
    while time.monotonic() < deadline:
        h = health(client, user)
        chunks = (h.get("retrieval") or {}).get("chunks")
        last = {"journal_pending": h.get("journal_pending"), "chunks": chunks}
        good = h.get("journal_pending") == 0 and (expect_chunks is None or chunks == expect_chunks)
        stable = stable + 1 if good else 0
        if stable >= 2:
            return {"ok": True, **last}
        sleep(2)
    return {"ok": False, **last}


def restore(client: Client, user: str, baseline: dict, env: dict | None = None, gate_log=None) -> dict:
    """Put the tenant back to `baseline`: PUT back every changed or missing file with its current version,
    and move every new file to Trash. Diary day files are protected from deletion (403), so a new day
    file is emptied instead: it then holds no entries and no chunks. Each write is gated (it reindexes).

    With env set, each write takes its own inference_slot. The lock is not re-entrant: calling this with
    env from INSIDE another inference_slot would wait on itself (until the lock timeout). Inside a slot,
    call it without env, as harness.restore_baseline does."""
    report = {"rewritten": [], "deleted": [], "emptied": [], "failures": []}
    for path in list_tree(client, user):
        if path in baseline:
            continue
        with inference_slot(env or {}, gate_log):
            st = client.request("POST", "/api/workspace-ops", user=user, json_body={"op": "stat", "path": path})
            version = (st.json() or {}).get("version")
            r = client.request("POST", "/api/workspace-ops", user=user,
                               json_body={"op": "delete", "path": path, "version": version}, timeout=300)
        if r.status == 403:
            with inference_slot(env or {}, gate_log):
                cur = client.request("POST", "/api/file", user=user, json_body={"path": path})
                body = cur.json() or {}
                if body.get("content") == "":
                    report["emptied"].append({"path": path, "status": 200})
                    continue
                r = client.request("PUT", "/api/file", user=user,
                                   json_body={"path": path, "content": "", "version": body.get("version")}, timeout=300)
            (report["emptied"] if r.status == 200 else report["failures"]).append({"path": path, "status": r.status})
            continue
        (report["deleted"] if r.status == 200 else report["failures"]).append({"path": path, "status": r.status})
    for path, content in sorted(baseline.items()):
        cur = client.request("POST", "/api/file", user=user, json_body={"path": path})
        body = cur.json() or {}
        if cur.status == 200 and body.get("content") == content:
            continue
        with inference_slot(env or {}, gate_log):
            r = client.request("PUT", "/api/file", user=user,
                               json_body={"path": path, "content": content, "version": body.get("version")}, timeout=300)
        (report["rewritten"] if r.status == 200 else report["failures"]).append({"path": path, "status": r.status})
    report["ok"] = not report["failures"]
    return report


def baseline_matches(client: Client, user: str, baseline: dict) -> bool:
    """The tenant equals the baseline, ignoring new files that restore() emptied (no text, no chunks)."""
    try:
        current = snapshot(client, user)
    except RuntimeError:
        return False
    return {p: t for p, t in current.items() if p in baseline or t != ""} == baseline


# ---------------- workflow-area result ----------------

class Area:
    """Collects checks for one workflow area and prints one JSON result: pass, fail, blocked or untested."""

    def __init__(self, name: str, run_dir, parallel_safe: bool):
        self.name, self.parallel_safe = name, parallel_safe
        self.dir = Path(run_dir) / "workflows" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.checks, self.requests = [], self.dir / "requests.jsonl"
        self.requests.write_text("", encoding="utf-8")

    def log_request(self, row: dict) -> None:
        append_jsonl(self.requests, row)

    def check(self, name: str, ok: bool, **detail) -> bool:
        self.checks.append({"check": name, "status": "pass" if ok else "fail", **detail})
        return ok

    def mark(self, name: str, status: str, reason: str) -> None:
        assert status in ("blocked", "untested")
        self.checks.append({"check": name, "status": status, "reason": reason})

    def status(self) -> str:
        found = {c["status"] for c in self.checks}
        for s in ("fail", "blocked", "untested"):
            if s in found:
                return s
        return "pass" if found else "untested"

    def finish(self, extra: dict | None = None) -> int:
        result = {"area": self.name, "status": self.status(), "parallel_safe": self.parallel_safe,
                  "counts": {s: sum(1 for c in self.checks if c["status"] == s) for s in STATUSES},
                  "checks": self.checks, **(extra or {})}
        path = write_json(self.dir / "result.json", result)
        result["evidence"] = [str(path), str(self.requests)]
        print(json.dumps({k: result[k] for k in ("area", "status", "parallel_safe", "counts", "evidence")}, sort_keys=True))
        return 0 if result["status"] in ("pass", "untested") else 1


def area_args(description: str, nomodel: bool = False):
    """nomodel areas default to the no-model diary, whose embed and chat URLs are disabled."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--base-url", default=NOMODEL_BASE_URL if nomodel else DEFAULT_BASE_URL)
    ap.add_argument("--env-file", default=str(DEFAULT_ENV))
    ap.add_argument("--run-dir", required=True, help="evidence run directory (harness.py init prints it)")
    ap.add_argument("--fixture", default=None, help="fixture directory from gen_fixture.py (areas that need the corpus)")
    ap.add_argument("--tenant-label", default=None, help="override the synthetic tenant label")
    return ap
