"""Shared helpers for the Diary gate: env file, signed tenant client, evidence layout, load gate.

Standard library only, so every gate script runs with `python3 -I` on the host and needs no venv.
Secrets (DIARY_AUTH_TOKEN, DIARY_TENANT_KEY, LLM_API_KEY) are read from .env.gate and are never
printed, logged or written to evidence.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

GATE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = GATE_DIR / ".env.gate"
DEFAULT_BASE_URL = "http://127.0.0.1:18910"
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


class Client:
    """Talks to the gate diary as the web server would: bearer token plus a signed tenant assertion."""

    def __init__(self, base_url: str, env: dict, timeout: float = 30.0, log=None):
        self.base_url = base_url.rstrip("/")
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
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
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


# ---------------- MS-01 load gate (before any inference call) ----------------

MIN_MEM_AVAILABLE = 6 * 1024 ** 3


def _get_json(url: str, token: str = "", timeout: float = 5.0):
    headers = {"Authorization": "Bearer " + token} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _slots_busy(base: str, token: str = "") -> tuple:
    """(busy, detail) from a chat-completions engine's /slots; a router needs ?model= for each loaded model."""
    base = base.rstrip("/")
    try:
        slots = _get_json(base + "/slots", token)
        if isinstance(slots, list):
            n = sum(1 for s in slots if isinstance(s, dict) and s.get("is_processing"))
            return n > 0, f"{n} slot(s) processing"
    except (OSError, ValueError, urllib.error.HTTPError):
        pass
    try:
        models = _get_json(base + "/v1/models", token).get("data") or []
    except (OSError, ValueError, AttributeError, urllib.error.HTTPError) as exc:
        return True, f"load unknown ({type(exc).__name__}); refusing to start"
    busy, seen = 0, 0
    for m in models:
        state = ((m.get("status") or {}).get("value") if isinstance(m, dict) else None)
        if state not in (None, "loaded"):
            continue
        try:
            slots = _get_json(base + "/slots?model=" + urllib.parse.quote(str(m.get("id"))), token)
        except (OSError, ValueError, urllib.error.HTTPError):
            continue
        if isinstance(slots, list):
            seen += 1
            busy += sum(1 for s in slots if isinstance(s, dict) and s.get("is_processing"))
    if not seen:
        return True, "no readable /slots; load unknown, refusing to start"
    return busy > 0, f"{busy} slot(s) processing across {seen} model(s)"


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def load_gate(env: dict) -> dict:
    """ok=False when MemAvailable < 6 GiB or the embed or chat engine has requests in flight (fail closed)."""
    result = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ok": True, "reasons": []}
    mem = mem_available()
    result["mem_available_gib"] = round(mem / 1024 ** 3, 2)
    if mem < MIN_MEM_AVAILABLE:
        result["ok"] = False
        result["reasons"].append(f"MemAvailable {result['mem_available_gib']} GiB < 6 GiB")
    for role, url_key, token_key in (("embed", "GATE_EMBED_LOAD_URL", ""), ("kev", "GATE_KEV_LOAD_URL", "LLM_API_KEY")):
        url = env.get(url_key, "")
        if not url:
            result["ok"] = False
            result["reasons"].append(f"{url_key} is not set; cannot check {role} load")
            continue
        busy, detail = _slots_busy(url, env.get(token_key, "") if token_key else "")
        result[role] = detail
        if busy:
            result["ok"] = False
            result["reasons"].append(f"{role}: {detail}")
    return result


# ---------------- fixture through the API ----------------

def fixture_files(fixture_dir: Path) -> list:
    """Corpus-relative paths of every fixture day file, in sorted order."""
    root = Path(fixture_dir) / "fixture"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.md"))


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_fixture(client: Client, user: str, fixture_dir: Path) -> dict:
    """PUT every day file through the workspace file API (version null = the file must not exist yet)."""
    root, failures, written = Path(fixture_dir) / "fixture", [], 0
    for rel in fixture_files(fixture_dir):
        content = (root / rel).read_text(encoding="utf-8")
        r = client.request("PUT", "/api/file", user=user, json_body={"path": rel, "content": content, "version": None},
                           timeout=120)
        if r.status == 200 and (r.json() or {}).get("version") == _sha(content.encode()):
            written += 1
        else:
            failures.append({"path": rel, "status": r.status})
    return {"written": written, "failures": failures}


def readback(client: Client, user: str, fixture_dir: Path, mutated: bool = False) -> dict:
    """Read every day back: file API byte-exact against the manifest (before mutations), month API with
    every xid marker present (after mutations the correction entries too)."""
    manifest = json.loads((Path(fixture_dir) / "manifest.json").read_text(encoding="utf-8"))
    entries = read_jsonl(Path(fixture_dir) / "truth" / "entries.jsonl")
    mismatched, months = [], {}
    for rel in fixture_files(fixture_dir):
        r = client.request("POST", "/api/file", user=user, json_body={"path": rel})
        content = (r.json() or {}).get("content") if r.status == 200 else None
        if content is None or (not mutated and _sha(content.encode()) != manifest["files"]["fixture/" + rel]):
            mismatched.append({"path": rel, "status": r.status})
    missing = []
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


def apply_mutations(client: Client, user: str, fixture_dir: Path) -> dict:
    """Scripted paraphrases and deletions (guarded edit by xid) and corrections (a new entry appended to a later day)."""
    results = []
    for m in read_jsonl(Path(fixture_dir) / "truth" / "mutations.jsonl"):
        if m["op"] == "edit":
            r = client.request("POST", "/api/entries/edit", user=user,
                               json_body={"xid": m["xid"], "me": m["me"], "assistant": m["assistant"], "month": m["month"]},
                               timeout=120)
            results.append({"id": m["id"], "status": r.status, "ok": r.status == 200})
        elif m["op"] == "add_entry":
            cur = client.request("POST", "/api/file", user=user, json_body={"path": m["path"]})
            body = cur.json() or {}
            if cur.status != 200 or body.get("content") is None:
                results.append({"id": m["id"], "status": cur.status, "ok": False})
                continue
            new = body["content"].rstrip("\n") + "\n\n" + m["block"] + "\n"
            r = client.request("PUT", "/api/file", user=user,
                               json_body={"path": m["path"], "content": new, "version": body.get("version")}, timeout=120)
            results.append({"id": m["id"], "status": r.status, "ok": r.status == 200})
    return {"applied": sum(1 for x in results if x["ok"]), "total": len(results), "results": results}


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


def area_args(description: str):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--base-url", default=os.environ.get("GATE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--env-file", default=str(DEFAULT_ENV))
    ap.add_argument("--run-dir", required=True, help="evidence run directory (harness.py init prints it)")
    ap.add_argument("--fixture", default=None, help="fixture directory from gen_fixture.py (areas that need the corpus)")
    ap.add_argument("--tenant-label", default=None, help="override the synthetic tenant label")
    return ap
