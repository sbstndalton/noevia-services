#!/usr/bin/env python3
"""Diary gate harness: load the synthetic fixture through the diary API, apply the scripted
mutations, run the ground-truth queries through /v1/chat/completions with signed tenant assertions,
and record answers and retrieval traces as evidence. score.py turns the evidence into stage results.

    harness.py init    --evidence-root DIR --rev REV --phase pre|post --fixture FIX
    harness.py load    --run-dir RUN --fixture FIX
    harness.py mutate  --run-dir RUN --fixture FIX          (then snapshots the post-mutation baseline)
    harness.py queries --run-dir RUN --fixture FIX [--only ID,...] [--limit N]
    harness.py all     --evidence-root DIR --rev REV --phase pre|post --fixture FIX

The fixture is verified against its manifest (every file hash and the tree hash) by every command.
With inference enabled, every write that reindexes (load, mutations, restores) and every query runs
inside gatelib.inference_slot: a cross-process lock around the MS-01 load check (bounded re-checks,
fail closed) and the call. After every query the tenant is restored to the post-mutation baseline
(changed files PUT back; new files moved to Trash, or emptied when they are protected diary day files)
and the index settles, so no query sees an earlier
query's logged exchange. Before each query turn and before each restore the harness waits (bounded,
GATE_IDLE_WAIT_S) for the engines to go idle, because a chat call the diary gave up on can keep generating
upstream; the wait is recorded (`idle_wait_s`) and a wait that runs out is "blocked", not a failed query.
Retrieval traces are recorded when the diary returns a known shape; otherwise each row says "trace unavailable".
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gatelib as g  # noqa: E402

CORPUS_LABEL = "gate-corpus"
TRACE_KEYS = ("trace", "retrieval_trace", "retrieval")
CANDIDATE_KEYS = ("candidates", "retrieved", "results")


def _client(args, run: Path, env: dict) -> g.Client:
    return g.Client(args.base_url, env, log=lambda row: g.append_jsonl(run / "logs" / "requests.jsonl", row))  # guarded URL


def _gate_log(run: Path, what: str):
    return lambda result: g.append_jsonl(run / "logs" / "load-gate.jsonl", {"for": what, **result})


def entry_index(fixture: Path) -> dict:
    index = {}
    for e in g.read_jsonl(fixture / "truth" / "entries.jsonl"):
        index[("xid", e["xid"])] = e["id"]
        index[("loc", e["path"], e["header"])] = e["id"]
        index[("dayhdr", e["day"], e["header"])] = e["id"]
        # Trace schema v1 ids are chunk rowids (not stable across reindexing): match by day and HH:MM.
        index[("daytime", e["day"], e["header"][:5])] = e["id"]
    return index


def extract_trace(payload: dict):
    """The diary's retrieval trace, if this revision returns one; None otherwise."""
    if not isinstance(payload, dict):
        return None
    diary = payload.get("diary") if isinstance(payload.get("diary"), dict) else {}
    for key in TRACE_KEYS:
        if isinstance(diary.get(key), (dict, list)):
            return diary[key]
        if isinstance(payload.get(key), (dict, list)):
            return payload[key]
    return None


def _ids(items, index: dict, exclude_xid: str | None, exclude_day: str | None = None) -> list:
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        xid = str(it.get("xid") or "")
        if (exclude_xid and xid == exclude_xid) or (exclude_day and str(it.get("day") or "") == exclude_day):
            continue  # this session's own logged exchanges (on the empty query day) are not candidates
        eid = (index.get(("xid", xid))
               or index.get(("daytime", str(it.get("day") or ""), str(it.get("time") or "")[:5]))
               or index.get(("loc", str(it.get("document") or it.get("path") or ""), str(it.get("header") or "")))
               or index.get(("dayhdr", str(it.get("day") or ""), str(it.get("header") or "").lstrip("# ").strip())))
        out.append(eid or f"unmapped:{str(it.get('header') or xid)[:40]}")
    return out


def normalize_trace(trace, index: dict, exclude_xid: str | None = None, exclude_day: str | None = None):
    """Map a trace onto entry ids: candidates, reranked, context, tokens. None for an unknown shape.

    Schema v1 (services#107): candidates [{rank, id, day, time, score, loaded, dropped}], loaded [{id, day,
    time, score, tokens, chars}], budget {max_context_tokens, ...}, rerank {present}."""
    if isinstance(trace, list):
        trace = {"candidates": trace}
    if not isinstance(trace, dict):
        return None
    if isinstance(trace.get("candidates"), list) and isinstance(trace.get("loaded"), list):
        cands = sorted((c for c in trace["candidates"] if isinstance(c, dict)), key=lambda c: c.get("rank") or 0)
        loaded = [x for x in trace["loaded"] if isinstance(x, dict)]
        budget = trace.get("budget") if isinstance(trace.get("budget"), dict) else {}
        rerank = trace.get("rerank") if isinstance(trace.get("rerank"), dict) else {}
        used = budget.get("used_tokens")
        if used is None and all(isinstance(x.get("tokens"), (int, float)) for x in loaded):
            used = sum(x["tokens"] for x in loaded)
        return {"schema": "v1", "status": trace.get("status"),
                "candidates": _ids(cands, index, exclude_xid, exclude_day),
                "reranked": _ids(rerank.get("order"), index, exclude_xid, exclude_day) if rerank.get("present") and isinstance(rerank.get("order"), list) else None,
                "context": _ids(loaded, index, exclude_xid, exclude_day), "context_tokens": used,
                "budget_tokens": budget.get("max_context_tokens")}
    cands = next((trace[k] for k in CANDIDATE_KEYS if isinstance(trace.get(k), list)), None)
    if cands is None:
        return None
    rer, ctx = trace.get("reranked"), trace.get("context") or trace.get("loaded") or trace.get("kept")
    return {"candidates": _ids(cands, index, exclude_xid),
            "reranked": _ids(rer, index, exclude_xid) if isinstance(rer, list) else None,
            "context": _ids(ctx, index, exclude_xid) if isinstance(ctx, list) else None,
            "context_tokens": trace.get("context_tokens") or trace.get("used_tokens"),
            "budget_tokens": trace.get("budget_tokens") or trace.get("max_context_tokens")}


def ask(client: g.Client, user: str, text: str, clock: str, session: str, history=None, timeout: float = 600):
    body = {"model": "diary", "messages": [*(history or []), {"role": "user", "content": text}], "session_id": session,
            "entryTime": clock.replace("Z", "+00:00")}
    started = time.monotonic()
    r = client.request("POST", "/v1/chat/completions", user=user, json_body=body, timeout=timeout,
                       extra_headers={"X-Diary-Retrieval-Trace": "1"})  # needs DIARY_RETRIEVAL_TRACE=1 (compose)
    payload = r.json() or {}
    answer = ""
    try:
        answer = payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    diary = (payload.get("diary") or {}) if isinstance(payload, dict) else {}
    return {"status": r.status, "answer": answer, "ms": round((time.monotonic() - started) * 1000),
            "decision": diary.get("decision"), "xid": diary.get("xid"), "trace": extract_trace(payload)}


def cmd_init(args) -> Path:
    fixture = Path(args.fixture)
    fman = g.verify_fixture(fixture)
    run = g.evidence_dir(args.evidence_root, args.rev, args.phase)
    shutil.copyfile(fixture / "manifest.json", run / "fixture-manifest.json")
    env = g.load_env(args.env_file)
    manifest = {"schema": "diarygate-run/1", "rev": args.rev, "phase": args.phase, "run_dir": str(run),
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "base_url": args.base_url,
                "inference": "on" if g.inference_enabled(env) else "off",
                "fixture": {"tree_sha256": fman["tree_sha256"], "seed": fman["seed"], "anchor": fman["anchor"],
                            "query_clock": fman["query_clock"], "generator_commit": fman["generator_commit"]},
                "thresholds_sha256": g._sha((g.GATE_DIR / "thresholds.json").read_bytes()),
                "rubric_sha256": g._sha((g.GATE_DIR / "rubric.json").read_bytes()),
                "layout": ["manifest.json", "matrix.csv", "stages/", "workflows/<area>/", "logs/", "blockers.md"]}
    g.write_json(run / "manifest.json", manifest)
    print(str(run))
    return run


def take_baseline(client: g.Client, user: str, env: dict, gate_log) -> dict:
    """Post-mutation baseline: every file's content plus (with inference) the settled chunk count."""
    with g.inference_slot(env, gate_log):
        settled = g.settle(client, user)
        files = g.snapshot(client, user)
        chunks = (g.health(client, user).get("retrieval") or {}).get("chunks")
    return {"ok": settled["ok"], "files": files, "chunks": chunks if g.inference_enabled(env) else None}


def restore_baseline(client: g.Client, user: str, baseline: dict, env: dict, gate_log) -> dict:
    """Wait for idle (an abandoned upstream generation would make the load gate refuse the writes), then
    restore, settle the index and verify, all inside ONE inference slot (the restore's writes and the
    recovery reindex embed). Raises GateBlocked when the engines never go idle or the load gate refuses."""
    # The idle wait runs BEFORE the inference lock is taken, so a long wait never holds up other gate
    # processes; inference_slot's require_gate then re-checks the load inside the lock.
    idle = g.require_idle(env, gate_log, "restore")
    with g.inference_slot(env, gate_log):
        report = g.restore(client, user, baseline["files"])
        settled = g.settle(client, user, expect_chunks=baseline.get("chunks"))
        same = g.baseline_matches(client, user, baseline["files"])
    report.update(settled=settled, matches=same, ok=report["ok"] and settled["ok"] and same, idle_wait_s=idle["waited_s"])
    return report


def _require_ok(run: Path, *names) -> None:
    for name in names:
        p = run / "stages" / f"{name}.json"
        if not p.exists() or not json.loads(p.read_text(encoding="utf-8")).get("ok"):
            raise SystemExit(f"refused: stages/{name}.json is missing or not ok")


def cmd_load(args, run: Path | None = None) -> dict:
    run = Path(run or args.run_dir)
    env, fixture = g.load_env(args.env_file), Path(args.fixture)
    g.verify_fixture(fixture)
    client, user, log = _client(args, run, env), g.tenant_id(args.tenant_label), _gate_log(run, "load")
    started = time.monotonic()
    try:
        loaded = g.load_fixture(client, user, fixture, env, log)
        back = g.readback(client, user, fixture, env=env, gate_log=log)
    except g.GateBlocked as exc:
        g.add_blocker(run, f"load stopped by the MS-01 load gate: {exc}")
        g.write_json(run / "stages" / "load.json", {"stage": "load", "ok": False, "blocked": str(exc)})
        print(json.dumps({"stage": "load", "ok": False, "blocked": str(exc)}))
        return {"ok": False}
    ok = not loaded["failures"] and not back["mismatched"] and not back["markers_missing"] and loaded["written"] > 0
    result = {"stage": "load", "ok": ok, "tenant": user, "seconds": round(time.monotonic() - started, 1), **loaded, "readback": back}
    g.write_json(run / "stages" / "load.json", result)
    if not ok:
        g.add_blocker(run, f"fixture load/read-back failed: {len(loaded['failures'])} write failures, "
                           f"{len(back['mismatched'])} mismatched files, {len(back['markers_missing'])} missing markers")
    print(json.dumps({"stage": "load", "ok": ok, "written": loaded["written"], "files_checked": back["files_checked"],
                      "mismatched": len(back["mismatched"]), "markers_missing": len(back["markers_missing"])}))
    return result


def cmd_mutate(args, run: Path | None = None) -> dict:
    run = Path(run or args.run_dir)
    env, fixture = g.load_env(args.env_file), Path(args.fixture)
    g.verify_fixture(fixture)
    _require_ok(run, "load")
    client, user, log = _client(args, run, env), g.tenant_id(args.tenant_label), _gate_log(run, "mutate")
    try:
        applied = g.apply_mutations(client, user, fixture, env, log)
        back = g.readback(client, user, fixture, mutated=True, env=env, gate_log=log)
        baseline = take_baseline(client, user, env, log)
    except g.GateBlocked as exc:
        g.add_blocker(run, f"mutations stopped by the MS-01 load gate: {exc}")
        g.write_json(run / "stages" / "mutations.json", {"stage": "mutate", "ok": False, "blocked": str(exc)})
        print(json.dumps({"stage": "mutate", "ok": False, "blocked": str(exc)}))
        return {"ok": False}
    ok = applied["applied"] == applied["total"] and not back["markers_missing"] and baseline["ok"]
    g.write_json(run / "stages" / "baseline.json", baseline)
    result = {"stage": "mutate", "ok": ok, "tenant": user, **applied, "markers_missing": back["markers_missing"],
              "baseline_files": len(baseline["files"]), "baseline_chunks": baseline["chunks"]}
    g.write_json(run / "stages" / "mutations.json", result)
    if not ok:
        g.add_blocker(run, f"scripted mutations: {applied['applied']}/{applied['total']} applied, baseline settled: {baseline['ok']}")
    print(json.dumps({"stage": "mutate", "ok": ok, "applied": applied["applied"], "total": applied["total"],
                      "baseline_files": len(baseline["files"])}))
    return result


def cmd_restore_selftest(args, run: Path | None = None) -> int:
    """Contaminate the tenant the way a logged exchange does (a new query-day file, an extra block on a
    fixture day), restore, and check the baseline is back byte for byte. Needs no inference."""
    run = Path(run or args.run_dir)
    env, fixture = g.load_env(args.env_file), Path(args.fixture)
    manifest = g.verify_fixture(fixture)
    _require_ok(run, "load", "mutations")
    baseline = json.loads((run / "stages" / "baseline.json").read_text(encoding="utf-8"))
    client, user, log = _client(args, run, env), g.tenant_id(args.tenant_label), _gate_log(run, "restore-selftest")
    from datetime import date
    qday = date.fromisoformat(manifest["query_clock"][:10])
    new = f"Entries/{qday.year}/{qday.strftime('%B')}/{qday.strftime('%B')} {qday.day}, {qday.year}.md"
    old = sorted(baseline["files"])[0]
    with g.inference_slot(env, log):
        w1 = client.request("PUT", "/api/file", user=user, json_body={"path": new, "content": "## Contamination\n\nlogged\n", "version": None})
        cur = client.request("POST", "/api/file", user=user, json_body={"path": old}).json() or {}
        w2 = client.request("PUT", "/api/file", user=user, json_body={"path": old, "content": (cur.get("content") or "") + "\nextra\n",
                                                                        "version": cur.get("version")})
    contaminated = w1.status == 200 and w2.status == 200 and not g.baseline_matches(client, user, baseline["files"])
    try:
        report = restore_baseline(client, user, baseline, env, log)
    except g.GateBlocked as exc:
        g.add_blocker(run, f"restore self-test stopped: {exc}")
        print(json.dumps({"stage": "restore-selftest", "ok": False, "blocked": str(exc)}))
        return 3
    ok = contaminated and report["ok"] and {d["path"] for d in report["deleted"] + report["emptied"]} == {new} and \
        {r["path"] for r in report["rewritten"]} == {old}
    g.write_json(run / "stages" / "restore-selftest.json", {"ok": ok, "contaminated": contaminated, **report})
    print(json.dumps({"stage": "restore-selftest", "ok": ok, "deleted": len(report["deleted"]), "emptied": len(report["emptied"]),
                      "rewritten": len(report["rewritten"]),
                      "matches": report["matches"]}))
    return 0 if ok else 1


def query_turns(q: dict) -> list:
    return [(q["text"], q)] + ([(q["followup"]["text"], q["followup"])] if q.get("followup") else [])


def run_turns(client, user, q, turns, env, gate_log, index, baseline, on_row, restore_log=None) -> str | None:
    """One query (all its turns), then the restore. Returns None, or a reason to stop the run."""
    history, session = [], f"gate-{q['id']}-{uuid.uuid4().hex[:8]}"
    for n, (text, spec) in enumerate(turns):
        if n and g.stop_requested():
            break  # SIGTERM: skip the follow-up turns, still restore below
        try:
            # Idle wait first, outside the inference lock (see restore_baseline); require_gate re-checks inside it.
            idle = g.require_idle(env, gate_log, f"{q['id']} turn {n}")
            with g.inference_slot(env, gate_log):
                res = ask(client, user, text, q["clock"], session, history)
            res["idle_wait_s"] = idle["waited_s"]
        except g.GateBlocked as exc:
            return f"blocked before {q['id']} turn {n}: MS-01 load gate: {exc}"
        qday = q["clock"][:10]
        empty_day = qday if not any(k[0] == "dayhdr" and k[1] == qday for k in index) else None
        trace = normalize_trace(res["trace"], index, res["xid"], empty_day) if res["trace"] is not None else None
        shape = "available" if trace else ("unknown shape" if res["trace"] is not None else "trace unavailable")
        on_row(n, text, spec, res, trace, shape)
        if res["status"] != 200:
            break
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": res["answer"]}]
    try:
        report = restore_baseline(client, user, baseline, env, gate_log)
    except g.GateBlocked as exc:
        return f"blocked at the restore after {q['id']}: MS-01 load gate: {exc}"
    if restore_log:
        restore_log({"query": q["id"], **{k: report[k] for k in ("deleted", "emptied", "rewritten", "failures", "settled", "matches", "ok",
                                                                  "idle_wait_s")}})
    if not report["ok"]:
        return f"restore after {q['id']} failed: {json.dumps({k: report[k] for k in ('failures', 'settled', 'matches')})}"
    return None


def cmd_queries(args, run: Path | None = None) -> int:
    run = Path(run or args.run_dir)
    env, fixture = g.load_env(args.env_file), Path(args.fixture)
    g.verify_fixture(fixture)
    if not g.inference_enabled(env):
        g.add_blocker(run, "queries not run: inference is disabled in .env.gate")
        print(json.dumps({"stage": "queries", "status": "blocked", "reason": "inference disabled"}))
        return 3
    _require_ok(run, "load", "mutations")
    baseline = json.loads((run / "stages" / "baseline.json").read_text(encoding="utf-8"))
    client, user, log = _client(args, run, env), g.tenant_id(args.tenant_label), _gate_log(run, "queries")
    index = entry_index(fixture)
    queries = g.read_jsonl(fixture / "truth" / "queries.jsonl")
    if args.only:
        wanted = set(args.only.split(","))
        queries = [q for q in queries if q["id"] in wanted]
    if args.limit:
        queries = queries[: args.limit]
    g.write_json(run / "stages" / "queries-plan.json", {"queries": [q["id"] for q in queries],
                                                         "planned_rows": sum(len(query_turns(q)) for q in queries)})
    answers, traces = run / "stages" / "answers.jsonl", run / "stages" / "traces.jsonl"
    for path in (answers, traces):
        path.write_text("", encoding="utf-8")
    done, idle_total = 0, [0.0]
    g.install_stop_handler()
    for q in queries:
        if g.stop_requested():
            stop = f"blocked: stopped by SIGTERM (area timeout) before {q['id']}; the tenant is restored"
            g.add_blocker(run, stop)
            print(json.dumps({"stage": "queries", "status": "blocked", "done": done, "reason": stop,
                              "idle_wait_s": round(idle_total[0], 3)}))
            return 3
        def on_row(n, text, spec, res, trace, shape, q=q):
            row_id = q["id"] if n == 0 else q["id"] + "#followup"
            idle_total[0] += res.get("idle_wait_s") or 0.0
            g.append_jsonl(answers, {"id": row_id, "query": q["id"], "turn": n, "category": q["category"], "status": res["status"],
                                     "answer": res["answer"], "ms": res["ms"], "decision": res["decision"], "trace": shape,
                                     "idle_wait_s": res.get("idle_wait_s"), "verdict": "needs-judge"})
            g.append_jsonl(traces, {"id": row_id, "trace": trace, "raw": res["trace"], "note": shape})
        stop = run_turns(client, user, q, query_turns(q), env, log, index, baseline, on_row,
                         lambda row: g.append_jsonl(run / "logs" / "restore.jsonl", row))
        if stop:
            g.add_blocker(run, stop)
            print(json.dumps({"stage": "queries", "status": "blocked", "done": done, "reason": stop,
                              "idle_wait_s": round(idle_total[0], 3)}))
            return 3
        done += 1
    print(json.dumps({"stage": "queries", "status": "done", "queries": done, "idle_wait_s": round(idle_total[0], 3)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("init", "load", "mutate", "restore-selftest", "queries", "all"))
    ap.add_argument("--base-url", default=g.DEFAULT_BASE_URL, help="default GATE_BASE_URL, else 127.0.0.1:GATE_PORT (18910)")
    ap.add_argument("--env-file", default=str(g.DEFAULT_ENV))
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--evidence-root")
    ap.add_argument("--rev")
    ap.add_argument("--phase", choices=("pre", "post"))
    ap.add_argument("--tenant-label", default=CORPUS_LABEL)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    g.check_base_url(args.base_url)  # refuse anything but a gate diary before any request
    if args.command in ("init", "all") and not (args.evidence_root and args.rev and args.phase):
        raise SystemExit("init/all need --evidence-root, --rev and --phase")
    if args.command not in ("init", "all") and not args.run_dir:
        raise SystemExit("--run-dir is required (harness.py init prints it)")
    if args.command == "init":
        cmd_init(args)
        return 0
    if args.command == "load":
        return 0 if cmd_load(args)["ok"] else 1
    if args.command == "mutate":
        return 0 if cmd_mutate(args)["ok"] else 1
    if args.command == "restore-selftest":
        return cmd_restore_selftest(args)
    if args.command == "queries":
        return cmd_queries(args)
    run = cmd_init(args)
    if not cmd_load(args, run)["ok"] or not cmd_mutate(args, run)["ok"]:
        return 1
    rc = cmd_queries(args, run)
    import score
    score.score(run, Path(args.fixture))
    return rc


if __name__ == "__main__":
    sys.exit(main())
