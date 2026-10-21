"""services#141: the tenant restore's writes share ONE total deadline (the write margin the stop grace counts),
a cut restore fails the area, and settle / the baseline check never run a full HTTP call past their deadline.
Fake HTTP servers on 127.0.0.1 only; small time constants; every test well under 10 s."""
import hashlib
import http.server
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / "gate"
sys.path.insert(0, str(GATE))
sys.path.insert(0, str(GATE / "areas"))

import gatelib as g  # noqa: E402
import harness  # noqa: E402
import run_areas  # noqa: E402
import score  # noqa: E402

SLACK = 1.0  # scheduling slack on top of a deadline


def v(t):
    return hashlib.sha256(t.encode()).hexdigest()


class Diary:
    """A tiny diary file API with configurable delays (seconds) per operation and optional hostile modes."""

    def __init__(self, files, put_delay=0.0, delete_delay=0.0, read_delay=0.0, list_delay=0.0, mode=None):
        self.files, self.mode = dict(files), mode
        self.put_delay, self.delete_delay, self.read_delay, self.list_delay = put_delay, delete_delay, read_delay, list_delay
        self.writes, self.lock = [], threading.Lock()


def _start(diary):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, payload):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass  # the client gave up (its timeout): fine

        def _hostile(self):
            if diary.mode == "5xx":
                self._send(503, b"{}")
                return True
            if diary.mode == "garbage":
                self._send(200, b"\xff\xfe not json at all")
                return True
            if diary.mode == "list":
                self._send(200, b"[1, 2, 3]")
                return True
            if diary.mode == "hang":
                time.sleep(3.0)  # never answers within the deadline; bounded so the server thread ends
                return True
            return False

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self._hostile():
                return
            if self.path.startswith("/api/files"):
                time.sleep(diary.list_delay)
                self._send(200, {"files": [{"path": p, "isDir": False} for p in sorted(diary.files)]})
            elif self.path.startswith("/api/health"):
                self._send(200, {"journal_pending": 0, "retrieval": {"chunks": 1}})
            else:
                self._send(404, {})

        def do_POST(self):
            body = self._body()
            if self._hostile():
                return
            if self.path == "/api/file":
                time.sleep(diary.read_delay)
                t = diary.files.get(body.get("path"))
                self._send(200, {"content": t, "version": v(t) if t is not None else None})
            elif self.path == "/api/workspace-ops" and body.get("op") == "stat":
                t = diary.files.get(body["path"])
                self._send(200, {"version": v(t) if t is not None else None})
            elif self.path == "/api/workspace-ops" and body.get("op") == "delete":
                with diary.lock:
                    diary.writes.append(("delete", body["path"]))
                time.sleep(diary.delete_delay)
                diary.files.pop(body["path"], None)
                self._send(200, {})
            else:
                self._send(404, {})

        def do_PUT(self):
            body = self._body()
            if self._hostile():
                return
            with diary.lock:
                diary.writes.append(("put", body["path"]))
            time.sleep(diary.put_delay)
            diary.files[body["path"]] = body["content"]
            self._send(200, {"version": v(body["content"])})

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def serve(monkeypatch):
    # The isolation guard (gate ports only) is not under test here: let the client reach the ephemeral fake port.
    monkeypatch.setattr(g, "check_base_url", lambda url, nomodel=False: url.rstrip("/"))
    servers = []

    def start(diary, client_timeout=5.0):
        srv = _start(diary)
        servers.append(srv)
        return g.Client(f"http://127.0.0.1:{srv.server_port}", {}, timeout=client_timeout)
    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


BASE = {f"Entries/2030/March/March {d}, 2030.md": f"day {d}\n" for d in range(1, 6)}


def _dirty(n):
    """The baseline with its first n files changed (each needs one PUT)."""
    files = dict(BASE)
    for p in sorted(BASE)[:n]:
        files[p] = files[p] + "logged exchange\n"
    return files


# ---------------- A: one total deadline, the same named constant the stop grace counts ----------------

def test_write_deadline_is_the_constant_the_grace_counts_as_the_write_margin(monkeypatch):
    assert g.RESTORE_WRITE_DEADLINE_S == run_areas.RESTORE_MARGIN_S == 300.0
    base = run_areas.area_grace_s({})
    monkeypatch.setattr(g, "RESTORE_WRITE_DEADLINE_S", g.RESTORE_WRITE_DEADLINE_S + 17.0)
    assert run_areas.area_grace_s({}) == base + 17.0  # change one and the other follows


def test_restore_uses_the_named_deadline_by_default(serve, monkeypatch):
    monkeypatch.setattr(g, "RESTORE_WRITE_DEADLINE_S", 0.5)
    d = Diary(_dirty(1), put_delay=3.0)
    c = serve(d)
    t0 = time.monotonic()
    report = g.restore(c, "u", BASE)
    assert time.monotonic() - t0 < 0.5 + SLACK
    assert report["ok"] is False and report["incomplete"] is True


def test_grace_with_120_rechecks_is_still_4500(monkeypatch):
    monkeypatch.setattr(g, "RECHECKS", 120)
    assert run_areas.area_grace_s({}) == 4500.0


# ---------------- F: the restore under a fake slow server ----------------

def test_restore_finishes_normally_within_the_deadline(serve):
    d = Diary({**_dirty(2), "Notes/new.md": "x"}, put_delay=0.05, delete_delay=0.05)
    report = g.restore(serve(d), "u", BASE, deadline_s=5.0)
    assert report["ok"] is True and report["incomplete"] is False and report["remaining"] == 0
    assert d.files == BASE and len(report["rewritten"]) == 2 and len(report["deleted"]) == 1


def test_one_slow_write_is_cut_at_the_deadline(serve):
    d = Diary(_dirty(1), put_delay=4.0)
    t0 = time.monotonic()
    report = g.restore(serve(d), "u", BASE, deadline_s=0.6)
    took = time.monotonic() - t0
    assert took < 0.6 + SLACK, took  # no single write runs past the total
    # The cut file plus the four baseline files the restore never got to verify.
    assert report["ok"] is False and report["incomplete"] is True and report["deadline_hit"] is True
    assert report["remaining"] == len(BASE) and len(d.writes) == 1


def test_many_slow_writes_stop_at_the_deadline_with_the_remaining_count(serve):
    d = Diary(_dirty(5), put_delay=0.5)
    t0 = time.monotonic()
    report = g.restore(serve(d), "u", BASE, deadline_s=1.2)
    took = time.monotonic() - t0
    assert took < 1.2 + SLACK, took
    assert report["ok"] is False and report["incomplete"] is True and report["deadline_hit"] is True
    issued = len(d.writes)
    assert 1 <= issued <= 3  # no further writes once the deadline is hit
    assert report["remaining"] == 5 - len(report["rewritten"]) and report["remaining"] >= 2


def test_unreachable_server_fails_the_restore_without_raising(monkeypatch):
    monkeypatch.setattr(g, "check_base_url", lambda url, nomodel=False: url.rstrip("/"))
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens there
    t0 = time.monotonic()
    report = g.restore(g.Client(f"http://127.0.0.1:{port}", {}, timeout=5.0), "u", BASE, deadline_s=1.0)
    assert time.monotonic() - t0 < 1.0 + SLACK
    assert report["ok"] is False and report["incomplete"] is True and report["remaining"] == len(BASE)


@pytest.mark.parametrize("mode", ["5xx", "garbage", "list", "hang"])
def test_hostile_responses_never_raise_or_overrun(serve, mode):
    d = Diary(_dirty(2), mode=mode)
    t0 = time.monotonic()
    report = g.restore(serve(d), "u", BASE, deadline_s=0.8)
    assert time.monotonic() - t0 < 0.8 + SLACK
    assert report["ok"] is False and report["incomplete"] is True and report["remaining"] >= 1


# ---------------- D: settle and the baseline check are clamped to their own deadline ----------------

def test_settle_never_runs_a_full_call_past_its_deadline(serve):
    c = serve(Diary(BASE, mode="hang"), client_timeout=5.0)
    t0 = time.monotonic()
    res = g.settle(c, "u", timeout=0.5)
    assert time.monotonic() - t0 < 0.5 + SLACK and res["ok"] is False


def test_baseline_check_is_clamped_to_its_deadline(serve):
    c = serve(Diary(BASE, read_delay=3.0), client_timeout=5.0)
    t0 = time.monotonic()
    assert g.baseline_matches(c, "u", BASE, deadline=time.monotonic() + 0.5) is False  # unverified is never a match
    assert time.monotonic() - t0 < 0.5 + SLACK


def test_restore_baseline_shares_one_post_write_budget_between_settle_and_check(serve, monkeypatch):
    monkeypatch.setattr(g, "RESTORE_WRITE_DEADLINE_S", 0.6)
    monkeypatch.setattr(g, "SETTLE_TIMEOUT_S", 0.6)
    c = serve(Diary(BASE, read_delay=3.0), client_timeout=5.0)
    t0 = time.monotonic()
    report = harness.restore_baseline(c, "u", {"files": BASE, "chunks": None}, {}, None)
    assert time.monotonic() - t0 < 0.6 + 0.6 + 2 * SLACK  # write deadline (reads cut) + the shared post-write budget
    assert report["ok"] is False and report["matches"] is False


# ---------------- B and C: a cut restore fails the area; the next query never runs on the dirty tenant ----------------

def _cut_restore(serve, monkeypatch):
    monkeypatch.setattr(g, "RESTORE_WRITE_DEADLINE_S", 0.5)
    monkeypatch.setattr(g, "SETTLE_TIMEOUT_S", 1.0)
    return serve(Diary(_dirty(3), put_delay=2.0))


def test_a_cut_restore_blocks_the_area_and_never_scores_as_pass(serve, monkeypatch, tmp_path):
    c = _cut_restore(serve, monkeypatch)
    asked = []
    monkeypatch.setattr(harness, "ask", lambda *a, **k: asked.append(a[2]) or {"status": 200, "answer": "a", "ms": 1,
                                                                               "decision": None, "xid": None, "trace": None})
    logged = []
    q1, q2 = {"id": "Q1", "clock": "2030-03-07T12:00:00Z"}, {"id": "Q2", "clock": "2030-03-07T12:00:00Z"}
    area = g.Area("semantic", tmp_path, parallel_safe=True)
    stop = None
    for q in (q1, q2):  # the loop _model._run_area, budgets.py and harness.cmd_queries all run
        stop = harness.run_turns(c, "u", q, [(q["id"], q)], {}, None, {}, {"files": BASE, "chunks": None},
                                 lambda *a: area.mark(f"{q['id']}#0", "untested", "needs-judge (rubric.json)"), logged.append)
        if stop:
            area.mark(q["id"], "blocked", stop)
            break
    assert asked == ["Q1"]  # C: Q2 never ran against the unrestored tenant
    assert stop.startswith("restore after Q1 failed") and '"incomplete": true' in stop and '"remaining": ' in stop
    assert logged[0]["ok"] is False and logged[0]["incomplete"] is True and logged[0]["remaining"] >= 1
    assert logged[0]["matches"] is False  # the baseline check confirms the tenant is not clean
    rc = area.finish()
    result = json.loads((area.dir / "result.json").read_text())
    assert result["status"] == "blocked" and rc == 1
    # B: judging every answer row "pass" still cannot turn the area into a pass.
    (area.dir / "answers.jsonl").write_text(json.dumps({"id": "semantic:Q1#0"}) + "\n")
    assert score.area_status(result, area.dir, {"semantic:Q1#0": {"verdict": "pass"}}) == "blocked"
