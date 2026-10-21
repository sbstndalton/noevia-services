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
    """A single engine (no router): one model, plain /slots."""
    if url.endswith("/v1/models"):
        return {"data": [{"id": "m"}]}
    if url.endswith("/slots"):
        return [{"is_processing": False}]
    raise AssertionError(url)


def test_load_gate_ok_when_idle_and_checks_aux_separately(tmp_path):
    seen = []
    r = g.load_gate(ENV, fetch=lambda u, t="": seen.append((u, t)) or idle(u), meminfo=meminfo(tmp_path, 8 * 1024 ** 2))
    assert r["ok"] and r["reasons"] == []
    assert all(t == "k" for _, t in seen)  # the token goes to the embed probe too
    assert {u for u, _ in seen if u.endswith("/slots")} == {"http://e/slots", "http://kev/slots"}  # aux is the chat engine here
    assert {u for u, _ in seen if u.endswith("/v1/models")} == {"http://e/v1/models", "http://kev/v1/models"}
    r = g.load_gate({**ENV, "LLM_AUX_BASE_URL": "http://aux/v1"}, fetch=lambda u, t="": seen.append((u, t)) or idle(u),
                    meminfo=meminfo(tmp_path, 8 * 1024 ** 2))
    assert "aux" in r and ("http://aux/slots", "k") in seen


@pytest.mark.parametrize("case", ["busy", "fetch-error", "low-mem", "no-meminfo", "no-url", "unreadable-model-slots"])
def test_load_gate_fails_closed(tmp_path, case):
    mem = meminfo(tmp_path, 8 * 1024 ** 2)
    env, fetch = dict(ENV), idle
    if case == "busy":
        fetch = lambda u, t="": {"data": [{"id": "m"}]} if u.endswith("/v1/models") else [{"is_processing": True}]  # noqa: E731
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
        if u.endswith("/v1/models"):
            return {"data": [{"id": "m1", "status": {"value": "loaded"}}, {"id": "m2", "status": {"value": "unloaded"}}]}
        if u.endswith("/slots?model=m1"):
            return [{"is_processing": False}]
        raise OSError("needs a loaded model")
    assert g._slots_busy("http://x", "", router)[0] is False

    def no_flag(u, t=""):  # a slot without is_processing counts as busy
        return {"data": [{"id": "m"}]} if u.endswith("/v1/models") else [{"id": 0}]
    assert g._slots_busy("http://x", "", no_flag)[0] is True
    assert g._slots_busy("http://x", "", lambda u, t="": {"data": []} if "models" in u else (_ for _ in ()).throw(OSError()))[0] is True


# ---------------- services#113 item 3: a load-gate probe never wakes a sleeping model ----------------

def _engine(models, slots=None, requested=None, routed=True):
    """A fake engine: /v1/models lists `models` ({id: state}); /slots?model=<id> answers only for ids in `slots`.
    Records every URL asked."""
    slots = slots or {}
    requested = requested if requested is not None else []

    def fetch(url, token=""):
        requested.append(url)
        if url.endswith("/v1/models"):
            return {"data": [({"id": i, "status": {"value": st}} if routed else {"id": i}) for i, st in models.items()]}
        if "/slots" in url:
            mid = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("model", [next(iter(models))])[0]
            if mid not in slots:
                raise AssertionError(f"slots requested for {mid}")
            return [{"is_processing": slots[mid]}]
        raise AssertionError(url)
    return fetch, requested


@pytest.mark.parametrize("models", [{"chat": "unloaded"}, {"chat": "unloaded", "embed": "unloaded"}])
def test_probe_never_requests_slots_when_the_model_list_says_unloaded(models, tmp_path):
    fetch, requested = _engine(models)
    busy, detail = g._slots_busy("http://router", "tok", fetch)
    assert busy is False and "unloaded" in detail and "not probed" in detail
    assert requested == ["http://router/v1/models"]  # the model list and nothing else
    # The whole load gate, every role: no URL ending in or containing /slots is ever requested.
    seen = []
    def fetch_all(url, token=""):
        seen.append(url)
        return _engine(models)[0](url, token)
    r = g.load_gate({**ENV, "LLM_AUX_BASE_URL": "http://aux/v1"}, fetch=fetch_all, meminfo=meminfo(tmp_path, 8 * 1024 ** 2))
    assert r["ok"] and r["reasons"] == []
    assert seen and all(u.endswith("/v1/models") for u in seen), seen
    assert all("/slots" not in u for u in seen)


def test_probe_asks_slots_only_for_the_models_listed_as_loaded():
    fetch, requested = _engine({"chat": "unloaded", "embed": "loaded", "rerank": "unloaded"}, slots={"embed": False})
    assert g._slots_busy("http://router", "", fetch) == (False, "2 model(s) unloaded, slots not probed; 0 slot(s) processing across 1 model(s)")
    assert requested == ["http://router/v1/models", "http://router/slots?model=embed"]
    fetch, requested = _engine({"chat": "loaded", "embed": "unloaded"}, slots={"chat": True})
    assert g._slots_busy("http://router", "", fetch)[0] is True  # loaded and generating
    assert [u for u in requested if "/slots" in u] == ["http://router/slots?model=chat"]


@pytest.mark.parametrize("state,why", [("loading", "loading"), ("sleeping", "not recognised")])
def test_probe_does_not_wake_a_model_that_is_loading_or_in_an_unknown_state(state, why):
    fetch, requested = _engine({"chat": state})
    busy, detail = g._slots_busy("http://router", "", fetch)
    assert busy is True and why in detail  # fails closed, still no slots request
    assert requested == ["http://router/v1/models"]


def test_probe_with_an_unreadable_model_list_refuses_without_asking_slots():
    seen = []
    def fetch(url, token=""):
        seen.append(url)
        raise OSError("down")
    busy, detail = g._slots_busy("http://router", "", fetch)
    assert busy is True and "model list unreadable" in detail
    assert seen == ["http://router/v1/models"]


def test_single_engine_without_a_router_uses_plain_slots_once_loaded():
    fetch, requested = _engine({"m": "loaded"}, slots={"m": False}, routed=False)
    assert g._slots_busy("http://engine", "", fetch)[0] is False
    assert requested == ["http://engine/v1/models", "http://engine/slots"]


def test_slots_endpoint_is_requested_from_one_helper_only():
    """Every probe call site is _slots_busy: no other function in the gate builds a /slots URL."""
    import ast
    sites = {}
    for path in sorted(GATE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and "/slots" in node.value:
                        sites.setdefault(path.name, set()).add(fn.name)
    assert sites == {"gatelib.py": {"_slots_busy"}}, sites


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


def _serve(handler_body):
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        hits = []

        def do_GET(self):
            H.hits.append(dict(self.headers))
            handler_body(self)

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, H


def _ok(h):
    h.send_response(200)
    h.send_header("Content-Type", "application/json")
    h.end_headers()
    h.wfile.write(b"[]")


def test_redirects_are_refused_and_never_followed():
    import urllib.request
    target, t = _serve(_ok)
    def redirect(h):
        h.send_response(302)
        h.send_header("Location", f"http://127.0.0.1:{target.server_port}/steal")
        h.end_headers()
    source, _ = _serve(redirect)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{source.server_port}/slots", headers={"Authorization": "Bearer secret"})
        with pytest.raises(g.RedirectRefused):
            g.OPENER.open(req, timeout=5)
        with pytest.raises(g.RedirectRefused):
            g._get_json(f"http://127.0.0.1:{source.server_port}/slots", "secret")
        assert t.hits == []  # the bearer token never reached the redirect target
    finally:
        source.shutdown()
        target.shutdown()


def test_proxy_settings_in_the_environment_are_ignored(monkeypatch):
    srv, h = _serve(_ok)
    for key in ("http_proxy", "HTTP_PROXY", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.setenv("no_proxy", "")
    try:
        assert g._get_json(f"http://127.0.0.1:{srv.server_port}/slots") == []
        assert len(h.hits) == 1
    finally:
        srv.shutdown()


@pytest.mark.parametrize("gate,nomodel,ok", [("18910", "18911", True), ("18912", "18919", True), ("11435", "18911", False),
                                             ("18080", "18911", False), ("18910", "18910", False), ("18920", "18911", False),
                                             ("8010", "18911", False), ("x", "18911", False)])
def test_gate_ports_are_pinned_to_the_gate_band(monkeypatch, gate, nomodel, ok):
    monkeypatch.setenv("GATE_PORT", gate)
    monkeypatch.setenv("GATE_NOMODEL_PORT", nomodel)
    if ok:
        assert g.gate_ports() == (int(gate), int(nomodel))
        assert g.check_base_url(f"http://127.0.0.1:{gate}")
    else:
        with pytest.raises(SystemExit):
            g.check_base_url(f"http://127.0.0.1:{gate}")
    assert not set(g.GATE_PORT_RANGE) & g.PRODUCTION_PORTS


# ---------------- services#113 item 1: model areas run one at a time; bounded waits report "blocked" ----------------

import run_areas  # noqa: E402


class FakeTime:
    """Stands in for gatelib's time module: sleeping advances the clock, nothing really waits."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __getattr__(self, name):
        import time
        return getattr(time, name)


def test_plan_runs_model_areas_one_at_a_time_and_no_model_areas_together():
    stages = run_areas.plan()
    assert stages[0] == ["capture", "tenancy"]  # no model: parallel
    assert stages[1:6] == [[a] for a in ("semantic", "temporal", "corrections", "follow_ups", "confirmed_vs_inferred")]
    assert stages[6:] == [["error_paths"], ["restart"], ["budgets"]]
    assert all(len(st) == 1 for st in stages[1:])  # nothing else runs beside a model area or an alone area
    assert run_areas.plan(parallel_model=True)[1] == list(run_areas.MODEL_AREAS)  # opt-in only
    assert run_areas.plan(only={"semantic", "capture"}) == [["capture"], ["semantic"]]


def _run_args(tmp_path, **kw):
    import argparse
    return argparse.Namespace(run_dir=str(tmp_path), fixture="fx", env_file="env", base_url=None, only="",
                              parallel_model_areas=False, **kw)


@pytest.mark.parametrize("parallel", [False, True])
def test_runner_executes_model_areas_serially_by_default(tmp_path, parallel):
    import threading
    import time
    lock, active, peak, started = threading.Lock(), set(), {}, []

    def runner(area, args, parallel_model):
        with lock:
            active.add(area)
            started.append(area)
            peak[area] = set(active)
        time.sleep(0.05)
        with lock:
            active.discard(area)
        return {"rc": 0, "out": json.dumps({"area": area, "status": "pass", "parallel_model": parallel_model})}

    args = _run_args(tmp_path)
    args.parallel_model_areas = parallel
    assert run_areas.run_all(args, runner=runner) == 0
    model = set(run_areas.MODEL_AREAS)
    if parallel:
        assert any(len(peak[a] & model) > 1 for a in model)  # the opt-in really overlaps
    else:
        assert all(peak[a] == {a} for a in model | set(run_areas.ALONE))  # no overlap at all
        assert started[2:7] == list(run_areas.MODEL_AREAS)  # semantic first, confirmed_vs_inferred last, in this order
    record = json.loads((tmp_path / "workflows" / "areas-run.json").read_text())
    assert record["parallel_model_areas"] is parallel and len(record["order"]) == 10
    assert record["order"][-3:] == ["error_paths", "restart", "budgets"]


def test_runner_continues_after_a_blocked_area_and_reports_nonzero(tmp_path, capsys):
    def runner(area, args, parallel_model):
        return {"rc": 1 if area == "temporal" else 0, "out": json.dumps({"area": area, "status": "blocked" if area == "temporal" else "pass"})}
    assert run_areas.run_all(_run_args(tmp_path), runner=runner) == 1
    ran = [json.loads(line)["area"] for line in capsys.readouterr().out.splitlines()]
    assert ran.index("corrections") > ran.index("temporal") and len(ran) == 10


def test_runner_children_get_the_parallel_switch_only_when_asked(tmp_path, monkeypatch):
    seen = []
    class Done:
        returncode, stdout, stderr = 0, '{"area": "x"}\n', ""
    monkeypatch.setenv("GATE_MODEL_AREAS_PARALLEL", "1")  # an inherited opt-in must not leak into a serial run
    monkeypatch.setattr(run_areas.subprocess, "run", lambda cmd, **kw: seen.append((cmd, kw["env"].get("GATE_MODEL_AREAS_PARALLEL"))) or Done())
    args = _run_args(tmp_path)
    run_areas.subprocess_runner("semantic", args, False)
    run_areas.subprocess_runner("semantic", args, True)
    assert [flag for _, flag in seen] == [None, "1"]
    cmd = seen[0][0]
    assert cmd[1] == "-I" and cmd[2].endswith("areas/semantic.py") and "--fixture" in cmd
    run_areas.subprocess_runner("capture", args, False)
    assert "--fixture" not in seen[-1][0]


def test_model_area_lock_blocks_a_second_area_with_the_holder_and_can_be_opted_out(tmp_path, monkeypatch):
    lock = str(tmp_path / "area.lock")
    logged = []
    monkeypatch.delenv("GATE_MODEL_AREAS_PARALLEL", raising=False)
    with g.model_area_slot("semantic", lock_file=lock) as first:
        assert first["serial"] is True
        with pytest.raises(g.GateBlocked) as exc:
            with g.model_area_slot("temporal", logged.append, lock_file=lock, timeout=0.2):
                pytest.fail("two model areas must not run at once")
        assert exc.value.result["status"] == "blocked" and exc.value.result["kind"] == "area-wait-timeout"
        reason = exc.value.result["reasons"][0]
        assert "temporal" in reason and "GATE_AREA_WAIT_S" in reason and f"pid={__import__('os').getpid()}" in reason
        assert logged[-1]["status"] == "blocked"
        monkeypatch.setenv("GATE_MODEL_AREAS_PARALLEL", "1")
        with g.model_area_slot("temporal", lock_file=lock, timeout=0.2) as second:
            assert second["serial"] is False
    monkeypatch.delenv("GATE_MODEL_AREAS_PARALLEL")
    with g.model_area_slot("temporal", lock_file=lock, timeout=0.2) as third:  # free again once the first area finished
        assert third["serial"] is True


def test_a_model_area_that_cannot_get_its_turn_reports_blocked_and_touches_nothing(fixture_dir, tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(GATE / "areas"))
    import contextlib
    import _model
    env = tmp_path / "env"
    env.write_text("LLM_BASE_URL=http://10.0.0.9:11435/v1\nDIARY_AUTH_TOKEN=t\n")
    monkeypatch.setattr(sys, "argv", ["semantic.py", "--run-dir", str(tmp_path / "run"), "--env-file", str(env), "--fixture", str(fixture_dir)])

    @contextlib.contextmanager
    def busy(name, log=None, **kw):
        exc = g.GateBlocked({"ok": False, "kind": "area-wait-timeout", "reasons": ["model-area lock still held by pid=1 corrections.py"]})
        log and log(exc.result)
        raise exc
        yield  # pragma: no cover
    monkeypatch.setattr(g, "model_area_slot", busy)
    monkeypatch.setattr(g, "Client", lambda *a, **k: pytest.fail("no request may be made while another model area runs"))
    assert _model.run("semantic", "doc", lambda q: True) == 1
    result = json.loads((tmp_path / "run" / "workflows" / "semantic" / "result.json").read_text())
    assert result["status"] == "blocked" and result["checks"][0]["check"] == "serial"
    assert "corrections.py" in result["checks"][0]["reason"]


def test_inference_lock_timeout_is_blocked_with_the_holder_and_wait(tmp_path):
    import fcntl
    import os
    fake = FakeTime()
    path = tmp_path / "inference.lock"
    with open(path, "a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        holder.write("pid=4242 semantic.py\n")
        holder.flush()
        fd = os.open(path, os.O_RDWR)
        with os.fdopen(fd, "r+") as fh:
            with pytest.raises(g.GateBlocked) as exc:
                g._acquire(fh, str(path), 3.0, sleep=fake.sleep, clock=fake.monotonic)
    r = exc.value.result
    assert r["status"] == "blocked" and r["kind"] == "lock-timeout" and r["lock_wait_s"] >= 3.0
    assert "pid=4242 semantic.py" in r["reasons"][0] and "GATE_LOCK_TIMEOUT_S" in r["reasons"][0]
    assert len(fake.sleeps) == 6 and set(fake.sleeps) == {0.5}  # bounded: polls every 0.5 s until the limit


def test_inference_slot_records_how_long_it_waited_for_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "require_gate", lambda env: {"ok": True, "reasons": []})
    logged = []
    with g.inference_slot(ENV, logged.append, lock_file=str(tmp_path / "lock")) as r:
        assert r["lock_wait_s"] >= 0
    assert "lock_wait_s" in logged[-1]


# ---------------- services#113 item 2: wait for idle before a restore and before the next query ----------------

def _busy_then_idle(busy_polls, clock):
    """engine_load fake: not ok for `busy_polls` polls, then ok."""
    polls = []

    def engine_load(env, fetch=None):
        polls.append(clock.now)
        if len(polls) <= busy_polls:
            return False, ["kev: 1 slot(s) processing across 1 model(s)"], {}
        return True, [], {}
    return engine_load, polls


def test_wait_for_idle_polls_until_the_abandoned_generation_ends(monkeypatch):
    clock = FakeTime()
    engine_load, polls = _busy_then_idle(3, clock)
    monkeypatch.setattr(g, "engine_load", engine_load)
    res = g.wait_for_idle(ENV, timeout=60, poll=5, sleep=clock.sleep, clock=clock.monotonic)
    assert res["idle"] is True and res["polls"] == 4 and res["waited_s"] == 15.0
    assert clock.sleeps == [5, 5, 5]


def test_wait_for_idle_is_bounded_and_the_last_sleep_never_overshoots(monkeypatch):
    clock = FakeTime()
    engine_load, polls = _busy_then_idle(10 ** 6, clock)
    monkeypatch.setattr(g, "engine_load", engine_load)
    res = g.wait_for_idle(ENV, timeout=12, poll=5, sleep=clock.sleep, clock=clock.monotonic)
    assert res["idle"] is False and res["waited_s"] == 12.0 and clock.sleeps == [5, 5, 2]
    assert "processing" in res["reasons"][0]


def test_wait_for_idle_does_nothing_without_inference(monkeypatch):
    monkeypatch.setattr(g, "engine_load", lambda *a, **k: pytest.fail("no probe when inference is disabled"))
    res = g.wait_for_idle({"LLM_BASE_URL": g.DISABLED_INFERENCE}, timeout=5)
    assert res["idle"] is True and res["waited_s"] == 0.0 and res["polls"] == 0


def test_wait_for_idle_probes_through_the_fake_http_without_waking_models():
    clock = FakeTime()
    states = iter([True, True, False])  # the abandoned generation ends on the third look
    requested = []

    def fetch_models(url):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "chat", "status": {"value": "loaded"}}, {"id": "other", "status": {"value": "unloaded"}}]}
        raise AssertionError(url)

    # The kev engine is the only one generating; embed is unloaded.
    def per_engine(url, token=""):
        requested.append(url)
        if url.startswith("http://e"):
            return {"data": [{"id": "emb", "status": {"value": "unloaded"}}]}
        if url.endswith("slots?model=chat"):
            return [{"is_processing": next(states)}]
        return fetch_models(url)
    res = g.wait_for_idle(ENV, timeout=60, poll=5, fetch=per_engine, sleep=clock.sleep, clock=clock.monotonic)
    assert res["idle"] is True and res["polls"] == 3 and res["waited_s"] == 10.0
    assert [u for u in requested if u.startswith("http://e")] == ["http://e/v1/models"] * 3  # embed is unloaded: models only
    assert all("model=other" not in u and not u.endswith("/slots") for u in requested)


def test_require_idle_records_the_wait_and_raises_blocked_when_it_never_ends(monkeypatch):
    clock = FakeTime()
    engine_load, _ = _busy_then_idle(2, clock)
    monkeypatch.setattr(g, "engine_load", engine_load)
    logged = []
    ok = g.require_idle(ENV, logged.append, "restore", timeout=60, poll=5, sleep=clock.sleep, clock=clock.monotonic)
    assert ok["waited_s"] == 10.0
    assert logged == [{"kind": "idle-wait", "before": "restore", "idle": True, "waited_s": 10.0, "polls": 3}]
    engine_load, _ = _busy_then_idle(10 ** 6, clock)
    monkeypatch.setattr(g, "engine_load", engine_load)
    with pytest.raises(g.GateBlocked) as exc:
        g.require_idle(ENV, logged.append, "next query", timeout=20, poll=5, sleep=clock.sleep, clock=clock.monotonic)
    r = exc.value.result
    assert r["status"] == "blocked" and r["kind"] == "not-idle" and r["idle_wait_s"] == 20.0
    assert "not idle after 20 s" in str(exc.value) and "GATE_IDLE_WAIT_S" in str(exc.value)
    assert logged[-1]["idle"] is False and logged[-1]["before"] == "next query" and logged[-1]["reasons"]


BASE_FILES = {"Entries/2030/January/January 6, 2030.md": "b\n"}


def _patch_restore_pieces(monkeypatch, events):
    monkeypatch.setattr(g, "restore", lambda c, u, files: events.append("restore") or {"ok": True, "deleted": [], "emptied": [],
                                                                                       "rewritten": [], "failures": []})
    monkeypatch.setattr(g, "settle", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(g, "baseline_matches", lambda *a, **k: True)
    monkeypatch.setattr(g, "require_gate", lambda env: {"ok": True, "reasons": []})


def test_restore_waits_for_idle_before_its_first_write(tmp_path, monkeypatch):
    clock = FakeTime()
    events = []
    engine_load, polls = _busy_then_idle(2, clock)
    monkeypatch.setattr(g, "engine_load", lambda env, fetch=None: events.append("poll") or engine_load(env, fetch))
    monkeypatch.setattr(g, "time", clock)
    _patch_restore_pieces(monkeypatch, events)
    monkeypatch.setattr(g, "_lock_path", lambda lock_file, name="inference.lock": str(tmp_path / "lock"))
    logged = []
    report = harness.restore_baseline(FakeDiary(BASE_FILES), "u", {"files": BASE_FILES, "chunks": None}, ENV, logged.append)
    assert events == ["poll", "poll", "poll", "restore"]  # every poll happened before the restore wrote anything
    assert report["ok"] and report["idle_wait_s"] == 10.0
    assert any(r.get("kind") == "idle-wait" and r["before"] == "restore" and r["waited_s"] == 10.0 for r in logged)


def test_restore_that_never_sees_idle_is_blocked_and_writes_nothing(tmp_path, monkeypatch):
    clock = FakeTime()
    events = []
    monkeypatch.setattr(g, "engine_load", lambda env, fetch=None: (False, ["kev: 1 slot(s) processing"], {}))
    monkeypatch.setattr(g, "time", clock)
    monkeypatch.setattr(g, "IDLE_WAIT_S", 30.0)
    _patch_restore_pieces(monkeypatch, events)
    with pytest.raises(g.GateBlocked) as exc:
        harness.restore_baseline(FakeDiary(BASE_FILES), "u", {"files": BASE_FILES, "chunks": None}, ENV, None)
    assert exc.value.result["kind"] == "not-idle" and events == []


QUERY = {"id": "Q1", "clock": "2030-03-07T12:00:00Z"}


def test_next_query_waits_for_idle_and_is_blocked_not_failed(tmp_path, monkeypatch):
    clock = FakeTime()
    order, rows, restores = [], [], []
    engine_load, _ = _busy_then_idle(2, clock)
    monkeypatch.setattr(g, "engine_load", lambda env, fetch=None: order.append("poll") or engine_load(env, fetch))
    monkeypatch.setattr(g, "time", clock)
    monkeypatch.setattr(g, "require_gate", lambda env: {"ok": True, "reasons": []})
    monkeypatch.setattr(g, "_lock_path", lambda lock_file, name="inference.lock": str(tmp_path / "lock"))
    monkeypatch.setattr(harness, "ask", lambda *a, **k: order.append("ask") or {"status": 200, "answer": "a", "ms": 1, "decision": None,
                                                                                 "xid": None, "trace": None})
    monkeypatch.setattr(harness, "restore_baseline", lambda *a, **k: order.append("restore") or
                        {"ok": True, "deleted": [], "emptied": [], "rewritten": [], "failures": [], "settled": {}, "matches": True,
                         "idle_wait_s": 0.0})
    stop = harness.run_turns(object(), "u", QUERY, [("hello", QUERY)], ENV, None, {}, {"files": {}},
                             lambda n, text, spec, res, trace, shape: rows.append(res), restores.append)
    assert stop is None and order == ["poll", "poll", "poll", "ask", "restore"]
    assert rows[0]["idle_wait_s"] == 10.0 and restores[0]["query"] == "Q1"

    # Never idle: the turn is not asked at all and the stop reason says blocked (no failed http row is produced).
    order.clear(), rows.clear()
    monkeypatch.setattr(g, "engine_load", lambda env, fetch=None: (False, ["kev: 1 slot(s) processing"], {}))
    monkeypatch.setattr(g, "IDLE_WAIT_S", 30.0)
    stop = harness.run_turns(object(), "u", QUERY, [("hello", QUERY)], ENV, None, {}, {"files": {}}, lambda *a: rows.append(a), None)
    assert stop.startswith("blocked before Q1 turn 0") and "not idle after 30 s" in stop
    assert order == [] and rows == []


def test_restore_after_an_abandoned_generation_is_blocked_while_the_answer_row_stays_recorded(tmp_path, monkeypatch):
    rows = []
    monkeypatch.setattr(g, "require_idle", lambda env, log=None, before="", **kw: {"idle": True, "waited_s": 0.0})
    monkeypatch.setattr(g, "inference_slot", lambda *a, **k: __import__("contextlib").nullcontext())
    monkeypatch.setattr(harness, "ask", lambda *a, **k: {"status": 504, "answer": "", "ms": 600000, "decision": None, "xid": None,
                                                         "trace": None})

    def refuse(*a, **k):
        raise g.GateBlocked({"ok": False, "kind": "not-idle", "reasons": ["engines not idle after 900 s"]})
    monkeypatch.setattr(harness, "restore_baseline", refuse)
    stop = harness.run_turns(object(), "u", QUERY, [("hello", QUERY)], ENV, None, {}, {"files": {}},
                             lambda n, text, spec, res, trace, shape: rows.append(res["status"]), None)
    assert rows == [504]  # the query's own timeout is recorded as it happened ...
    assert stop.startswith("blocked at the restore after Q1") and "not idle" in stop  # ... the restore is blocked, not failed
