"""Diary gate tooling (diary/gate): load gate, restore, fixture verification, scoring and isolation."""
import hashlib
import json
import sys
import urllib.parse
from datetime import date
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / "gate"
sys.path.insert(0, str(GATE))

import gatelib as g  # noqa: E402
import gen_fixture  # noqa: E402
import harness  # noqa: E402
import isolation  # noqa: E402
import score  # noqa: E402

ENV = {"LLM_BASE_URL": "http://10.0.0.9:11435/v1", "LLM_API_KEY": "k", "GATE_EMBED_LOAD_URL": "http://e",
       "GATE_KEV_LOAD_URL": "http://kev", "LLM_AUX_BASE_URL": "http://kev/v1"}


def meminfo(tmp_path, kib):
    p = tmp_path / "meminfo"
    p.write_text(f"MemTotal: 1 kB\nMemAvailable: {kib} kB\n")
    return str(p)


def idle(url, token=""):
    if url.endswith("/slots"):
        return [{"is_processing": False}]
    raise AssertionError(url)


def test_load_gate_ok_when_idle_and_checks_aux_separately(tmp_path):
    seen = []
    r = g.load_gate(ENV, fetch=lambda u, t="": seen.append((u, t)) or idle(u), meminfo=meminfo(tmp_path, 8 * 1024 ** 2))
    assert r["ok"] and r["reasons"] == []
    assert all(t == "k" for _, t in seen)  # the token goes to the embed probe too
    assert {u for u, _ in seen} == {"http://e/slots", "http://kev/slots"}  # aux is the chat engine here
    r = g.load_gate({**ENV, "LLM_AUX_BASE_URL": "http://aux/v1"}, fetch=lambda u, t="": seen.append((u, t)) or idle(u),
                    meminfo=meminfo(tmp_path, 8 * 1024 ** 2))
    assert "aux" in r and ("http://aux/slots", "k") in seen


@pytest.mark.parametrize("case", ["busy", "fetch-error", "low-mem", "no-meminfo", "no-url", "unreadable-model-slots"])
def test_load_gate_fails_closed(tmp_path, case):
    mem = meminfo(tmp_path, 8 * 1024 ** 2)
    env, fetch = dict(ENV), idle
    if case == "busy":
        fetch = lambda u, t="": [{"is_processing": True}]  # noqa: E731
    elif case == "fetch-error":
        def fetch(u, t=""):
            raise OSError("refused")
    elif case == "low-mem":
        mem = meminfo(tmp_path, 5 * 1024 ** 2)
    elif case == "no-meminfo":
        mem = str(tmp_path / "missing")
    elif case == "no-url":
        env["GATE_EMBED_LOAD_URL"] = ""
    else:
        def fetch(u, t=""):
            if u.endswith("/v1/models"):
                return {"data": [{"id": "a", "status": {"value": "loaded"}}, {"id": "b"}]}
            if u.endswith("model=a"):
                return [{"is_processing": False}]
            raise OSError("no slots")
    r = g.load_gate(env, fetch=fetch, meminfo=mem)
    assert r["ok"] is False and r["reasons"]


def test_slots_busy_router_and_missing_flag():
    def router(u, t=""):
        if u.endswith("/slots"):
            raise OSError("needs model")
        if u.endswith("/v1/models"):
            return {"data": [{"id": "m1", "status": {"value": "loaded"}}, {"id": "m2", "status": {"value": "unloaded"}}]}
        return [{"is_processing": False}]
    assert g._slots_busy("http://x", "", router)[0] is False
    assert g._slots_busy("http://x", "", lambda u, t="": [{"id": 0}])[0] is True  # no is_processing flag: busy
    assert g._slots_busy("http://x", "", lambda u, t="": {"data": []} if "models" in u else (_ for _ in ()).throw(OSError()))[0] is True


def test_require_gate_rechecks_then_blocks():
    calls, sleeps = [], []
    seq = iter([False, False, True])
    r = g.require_gate({}, rechecks=5, wait=1, check=lambda e: calls.append(1) or {"ok": next(seq), "reasons": ["busy"]},
                       sleep=sleeps.append)
    assert r["ok"] and r["attempt"] == 3 and len(sleeps) == 2
    with pytest.raises(g.GateBlocked) as exc:
        g.require_gate({}, rechecks=3, wait=1, check=lambda e: {"ok": False, "reasons": ["kev: busy"]}, sleep=sleeps.append)
    assert exc.value.result["attempt"] == 3 and "kev: busy" in str(exc.value)


def test_inference_slot_is_noop_without_inference_and_locks_with_it(tmp_path, monkeypatch):
    with g.inference_slot({"LLM_BASE_URL": g.DISABLED_INFERENCE}) as r:
        assert r is None
    logged = []
    monkeypatch.setattr(g, "require_gate", lambda env: {"ok": True, "reasons": []})
    with g.inference_slot(ENV, logged.append, lock_file=str(tmp_path / "lock")) as r:
        assert r["ok"]
    assert logged and (tmp_path / "lock").exists()
    def refuse(env):
        raise g.GateBlocked({"ok": False, "reasons": ["embed: 1 slot(s) processing"]})
    monkeypatch.setattr(g, "require_gate", refuse)
    with pytest.raises(g.GateBlocked):
        with g.inference_slot(ENV, logged.append, lock_file=str(tmp_path / "lock")):
            pytest.fail("the guarded call must not run")
    assert logged[-1]["ok"] is False


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tooling") / "fix"
    gen_fixture.write(out, 20261020, date(2030, 1, 6), 60, "UTC", "0" * 40)
    return out


def test_verify_fixture_refuses_unlisted_changed_and_missing(fixture_dir, tmp_path):
    import shutil
    assert g.verify_fixture(fixture_dir)["tree_sha256"]
    for case in ("unlisted", "changed-truth", "missing", "no-manifest"):
        bad = tmp_path / case
        shutil.copytree(fixture_dir, bad)
        if case == "unlisted":
            (bad / "fixture" / "Entries" / "extra.md").write_text("x")
        elif case == "changed-truth":
            with open(bad / "truth" / "queries.jsonl", "a") as fh:
                fh.write("{}\n")
        elif case == "missing":
            (bad / "truth" / "mutations.jsonl").unlink()
        else:
            (bad / "manifest.json").unlink()
        with pytest.raises(SystemExit):
            g.verify_fixture(bad)


class FakeDiary:
    """Enough of the file and workspace-ops API for restore()."""

    def __init__(self, files):
        self.files = dict(files)

    def request(self, method, path, user=None, query="", json_body=None, **kw):
        body = json_body or {}
        v = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
        if path == "/api/files":
            base = urllib.parse.unquote(query[5:]) if query.startswith("path=") else ""
            names = {}
            for p in self.files:
                if base and not p.startswith(base + "/"):
                    continue
                rest = p[len(base) + 1:] if base else p
                head = rest.split("/", 1)[0]
                names[head] = "/" in rest
            out = [{"path": f"{base}/{n}" if base else n, "isDir": d} for n, d in sorted(names.items())]
            return g.Response(200, json.dumps({"files": out}).encode(), {})
        if path == "/api/file" and method == "POST":
            t = self.files.get(body["path"])
            return g.Response(200, json.dumps({"content": t, "version": v(t) if t is not None else None}).encode(), {})
        if path == "/api/file" and method == "PUT":
            cur = self.files.get(body["path"])
            if (v(cur) if cur is not None else None) != body["version"]:
                return g.Response(409, b"{}", {})
            self.files[body["path"]] = body["content"]
            return g.Response(200, json.dumps({"version": v(body["content"])}).encode(), {})
        if path == "/api/workspace-ops":
            if body["op"] == "stat":
                return g.Response(200, json.dumps({"version": v(self.files[body["path"]])}).encode(), {})
            assert body["version"] == v(self.files[body["path"]])
            if body["path"].startswith("Entries/"):
                return g.Response(403, b"{}", {})  # diary day files are protected (agent/workspace_ops.py)
            del self.files[body["path"]]
            return g.Response(200, b"{}", {})
        raise AssertionError((method, path))


def test_restore_puts_back_changed_files_and_deletes_new_ones():
    base = {"Entries/2030/March/March 6, 2030.md": "a\n", "Entries/2030/January/January 6, 2030.md": "b\n"}
    d = FakeDiary(base)
    d.files["Entries/2030/March/March 7, 2030.md"] = "logged exchange\n"
    d.files["Entries/2030/January/January 6, 2030.md"] = "b\nlogged\n"
    d.files["Notes/scratch.md"] = "new note"
    report = g.restore(d, "u", base)
    assert report["ok"] and d.files == {**base, "Entries/2030/March/March 7, 2030.md": ""}
    assert [x["path"] for x in report["deleted"]] == ["Notes/scratch.md"]
    assert [x["path"] for x in report["emptied"]] == ["Entries/2030/March/March 7, 2030.md"]
    assert [x["path"] for x in report["rewritten"]] == ["Entries/2030/January/January 6, 2030.md"]
    assert g.baseline_matches(d, "u", base)


def test_normalize_trace_v1_matches_by_day_and_time(fixture_dir):
    index = harness.entry_index(fixture_dir)
    e = g.read_jsonl(fixture_dir / "truth" / "entries.jsonl")[0]
    trace = {"params": {"top_k": 8}, "status": "ok", "rerank": {"present": False},
             "candidates": [{"rank": 2, "id": 99, "day": "2030-03-07", "time": "12:00", "score": 0.9},
                            {"rank": 1, "id": 5, "day": e["day"], "time": e["header"][:5], "score": 0.8}],
             "loaded": [{"id": 5, "day": e["day"], "time": e["header"][:5], "tokens": 40, "chars": 150}],
             "budget": {"max_context_tokens": 5200}, "counts": {}, "direct_fallback": False}
    t = harness.normalize_trace(trace, index, None, "2030-03-07")
    assert t["candidates"] == [e["id"]] and t["context"] == [e["id"]] and t["reranked"] is None
    assert t["context_tokens"] == 40 and t["budget_tokens"] == 5200
    assert harness.normalize_trace({"something": []}, index) is None
    assert harness.normalize_trace("text", index) is None


TH = json.loads((GATE / "thresholds.json").read_text())


def _rows(n_hit, n_total, cands_hit):
    queries = {f"q{i}": {"relevant_entry_ids": [[f"E{i}"]]} for i in range(n_total)}
    traces = [{"id": f"q{i}", "trace": {"candidates": cands_hit(i) if i < n_hit else ["X1", "X2"], "reranked": None,
                                        "context": None}} for i in range(n_total)]
    return queries, traces


def test_stage_retrieval_threshold_boundaries():
    q, t = _rows(9, 10, lambda i: [f"E{i}", "X"])           # recall 0.90, precision 0.45
    cand, rr, ctx = score.stage_retrieval(q, t, TH)
    assert cand["recall@8"] == 0.9 and cand["status"] == "pass"
    assert rr["status"] == "N/A: no reranker in diary" and ctx["status"].startswith("untested")
    q, t = _rows(8, 10, lambda i: [f"E{i}"])                # recall 0.80
    assert score.stage_retrieval(q, t, TH)[0]["status"] == "fail"
    q = {f"q{i}": {"relevant_entry_ids": [[f"E{i}", f"F{i}"]]} for i in range(5)}
    t = [{"id": f"q{i}", "trace": {"candidates": [f"E{i}", f"F{i}", "a", "b", "c"]}} for i in range(5)]
    assert score.stage_retrieval(q, t, TH)[0]["precision@8"] == 0.4
    assert score.stage_retrieval(q, t, TH)[0]["status"] == "pass"   # exactly at 0.40
    t = [{"id": f"q{i}", "trace": {"candidates": [f"E{i}", f"F{i}", "a", "b", "c", "d"]}} for i in range(5)]
    assert score.stage_retrieval(q, t, TH)[0]["status"] == "fail"   # 0.33
    assert score.stage_retrieval(q, [{"id": "q0", "trace": None}], TH)[0]["status"].startswith("untested")


def _full_run(tmp_path, fixture_dir, area_status="pass"):
    run = g.evidence_dir(tmp_path / "ev", "d" * 40, "post")
    g.write_json(run / "stages" / "load.json", {"ok": True})
    g.write_json(run / "stages" / "mutations.json", {"ok": True})
    queries = g.read_jsonl(fixture_dir / "truth" / "queries.jsonl")
    plan_rows, judgements = 0, []
    for q in queries:
        for n, (text, spec) in enumerate(harness.query_turns(q)):
            rid = q["id"] if n == 0 else q["id"] + "#followup"
            plan_rows += 1
            g.append_jsonl(run / "stages" / "answers.jsonl", {"id": rid, "query": q["id"], "turn": n, "category": q["category"],
                                                               "status": 200, "answer": "x", "verdict": "needs-judge"})
            rel = sorted({i for s in spec.get("relevant_entry_ids", []) for i in s})
            g.append_jsonl(run / "stages" / "traces.jsonl", {"id": rid, "trace": {"candidates": rel, "reranked": None,
                                                                                 "context": rel, "context_tokens": 100}})
            abstain = bool(spec.get("expect_abstain")) and n == 0
            judgements.append({"row": rid, "verdict": "pass", "fact": None if abstain else True, "citation": True,
                               "stale": False, "abstained": abstain})
    g.write_json(run / "stages" / "queries-plan.json", {"queries": [q["id"] for q in queries], "planned_rows": plan_rows})
    g.write_json(run / "workflows" / "capture" / "result.json", {"area": "capture", "status": area_status, "parallel_safe": True,
                                                                 "checks": []})
    return run, judgements


def test_verdict_is_never_pass_with_needs_judge_rows_and_passes_once_judged(fixture_dir, tmp_path):
    run, judgements = _full_run(tmp_path, fixture_dir)
    result = score.score(run, fixture_dir)
    stages = {s["stage"]: s["status"] for s in result["stages"]}
    assert stages["answer"] == "needs-judge" and stages["reranking"] == "N/A: no reranker in diary"
    assert stages["candidate_retrieval"] == "pass" and result["verdict"] != "pass" and result["blended_score"] is None
    path = run / "judgements.jsonl"
    for j in judgements[:-1]:
        g.append_jsonl(path, j)
    assert score.score(run, fixture_dir)["verdict"] != "pass"  # one row still needs a judge
    g.append_jsonl(path, judgements[-1])
    result = score.score(run, fixture_dir)
    assert {s["stage"]: s["status"] for s in result["stages"]}["answer"] == "pass" and result["verdict"] == "pass"
    g.append_jsonl(path, {**judgements[0], "verdict": "fail", "stale": True})
    assert score.score(run, fixture_dir)["verdict"] != "pass"


def test_partial_runs_and_unready_runs_are_untested(fixture_dir, tmp_path):
    run, _ = _full_run(tmp_path, fixture_dir)
    lines = (run / "stages" / "answers.jsonl").read_text().splitlines()
    (run / "stages" / "answers.jsonl").write_text("\n".join(lines[:10]) + "\n")
    stages = {s["stage"]: s["status"] for s in score.score(run, fixture_dir)["stages"]}
    assert stages["answer"] == f"untested: partial 10/{len(lines)}" and stages["candidate_retrieval"].startswith("untested: partial")
    g.write_json(run / "stages" / "mutations.json", {"ok": False})
    stages = {s["stage"]: s["status"] for s in score.score(run, fixture_dir)["stages"]}
    assert stages["mutations"] == "fail" and stages["answer"] == "untested: load or mutations not ok"


def _inspect(name, **over):
    info = {"Name": "/" + name, "Mounts": [{"Type": "volume", "Name": "diarygate_data", "Destination": "/data"}],
            "NetworkSettings": {"Networks": {"diarygate_net": {}}},
            "Config": {"Image": "diarygate-diary:abc", "Env": ["DIARY_AUTH_TOKEN=s", "TZ=UTC"]},
            "HostConfig": {"Binds": ["diarygate_data:/data:rw"], "PortBindings": {"8010/tcp": [{"HostIp": "127.0.0.1", "HostPort": "1"}]}}}
    if name == isolation.NOMODEL:
        info["Config"]["Env"] += [f"{k}=http://gate-inference-disabled.invalid:9/v1" for k in isolation.NOMODEL_URL_KEYS]
    for k, v in over.items():
        info[k] = {**info[k], **v} if isinstance(v, dict) else v
    return info


@pytest.mark.parametrize("bad", [None, "bind", "network", "port", "env", "nomodel-url", "image", "missing", "volume"])
def test_isolation_check_with_fake_inspect(bad):
    a, b = _inspect("diarygate-diary"), _inspect(isolation.NOMODEL)
    vols, nets = ["diarygate_data"], ["diarygate_net"]
    if bad == "bind":
        a["HostConfig"]["Binds"] = ["/srv/appdata:/data"]
    elif bad == "network":
        a["NetworkSettings"]["Networks"]["bridge"] = {}
    elif bad == "port":
        a["HostConfig"]["PortBindings"]["8010/tcp"][0]["HostIp"] = "0.0.0.0"
    elif bad == "env":
        a["Config"]["Env"].append("WEBDAV_PASSWORD=x")
    elif bad == "nomodel-url":
        b["Config"]["Env"] = [e for e in b["Config"]["Env"] if not e.startswith("LLM_EMBED_BASE_URL")] + ["LLM_EMBED_BASE_URL=http://10.0.0.9:18080/v1"]
    elif bad == "image":
        a["Config"]["Image"] = "cowork-diary:live"
    elif bad == "volume":
        vols.append("cowork_diary_data")
    containers = [a] if bad == "missing" else [a, b]
    result = isolation.check_containers(containers, vols, nets)
    assert result["ok"] is (bad is None), result["problems"]


def test_http_failure_on_a_turn_with_a_followup_is_fail_not_untested(fixture_dir, tmp_path):
    run, _ = _full_run(tmp_path, fixture_dir)
    path = run / "stages" / "answers.jsonl"
    rows = g.read_jsonl(path)
    first = next(r for r in rows if r["id"] + "#followup" in {x["id"] for x in rows})
    kept = [dict(r, status=502) if r["id"] == first["id"] else r for r in rows if r["id"] != first["id"] + "#followup"]
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    stages = {s["stage"]: s["status"] for s in score.score(run, fixture_dir)["stages"]}
    assert stages["answer"] == "fail" and stages["candidate_retrieval"].startswith("untested: partial")


@pytest.mark.parametrize("url,nomodel,ok", [
    ("http://127.0.0.1:18910", False, True), ("http://localhost:18911/", False, True), ("http://127.0.0.1:18911", True, True),
    ("http://127.0.0.1:18910", True, False), ("http://10.69.0.131:18910", False, False), ("http://127.0.0.1:11435", False, False),
    ("https://127.0.0.1:18910", False, False), ("http://u:p@127.0.0.1:18910", False, False), ("http://127.0.0.1", False, False),
    ("http://127.0.0.1:18910/v1", False, False), ("http://127.0.0.1.example:18910", False, False), ("not a url", False, False)])
def test_base_url_guard(url, nomodel, ok):
    if ok:
        assert g.check_base_url(url, nomodel).startswith("http://")
        g.Client(url, {}, nomodel=nomodel)
    else:
        with pytest.raises(SystemExit):
            g.Client(url, {}, nomodel=nomodel)


def test_no_model_area_refuses_the_gate_diary(tmp_path, monkeypatch, capsys):
    import runpy
    monkeypatch.setattr(sys, "argv", ["capture.py", "--run-dir", str(tmp_path), "--env-file", str(tmp_path / "env"),
                                      "--base-url", "http://127.0.0.1:18910"])
    (tmp_path / "env").write_text("DIARY_AUTH_TOKEN=t\n")
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(GATE / "areas" / "capture.py"), run_name="__main__")
    assert "no-model diary" in str(exc.value)


def test_inference_lock_times_out_fail_closed_in_a_private_dir(tmp_path, monkeypatch):
    import fcntl
    import os
    monkeypatch.setattr(g, "require_gate", lambda env: {"ok": True, "reasons": []})
    monkeypatch.setattr(g, "LOCK_DIR", str(tmp_path / "locks"))
    with g.inference_slot(ENV):
        pass
    assert (os.stat(tmp_path / "locks").st_mode & 0o777) == 0o700
    lock = str(tmp_path / "locks" / "inference.lock")
    with open(lock, "a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        logged = []
        with pytest.raises(g.GateBlocked):
            with g.inference_slot(ENV, logged.append, lock_timeout=0.2):
                pytest.fail("must not run while another process holds the lock")
        assert "lock still held" in logged[-1]["reasons"][0]
