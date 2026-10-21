"""Shared runner for the workflow areas that need inference (semantic, temporal, corrections,
follow-ups, confirmed-vs-inferred). Each area loads the verified fixture into its OWN tenant, so these
areas are parallel-safe with each other. Every write batch and every inference call runs inside
gatelib.inference_slot (cross-process lock, bounded re-checks, fail closed), and after each query the
tenant is restored to its post-mutation baseline exactly as harness.py does.

Model areas run ONE AT A TIME by default (gatelib.model_area_slot: a cross-process lock held for the whole
area, bounded by GATE_AREA_WAIT_S, "blocked" with the holder when it runs out; run_areas.py orders them
the same way). GATE_MODEL_AREAS_PARALLEL=1 opts back in to running them together. Each query turn and each
restore first waits (bounded, GATE_IDLE_WAIT_S) for the engines to go idle; the total is in result.json as
idle_wait_s.

Without inference (smoke mode) they report "blocked". With inference, answer rows are recorded with
deterministic rubric signals and stay "needs-judge" (status "untested") unless a request fails;
score.py applies judgements.jsonl to them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GATE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GATE))
import gatelib as g  # noqa: E402
import harness  # noqa: E402
import score  # noqa: E402


def default_turns(q: dict) -> list:
    return [(q["text"], q)]


def run(name: str, doc: str, select, turns=default_turns, mutate: bool = True) -> int:
    args = g.area_args(doc).parse_args()
    area = g.Area(name, args.run_dir, True)
    env = g.load_env(args.env_file)
    if not g.inference_enabled(env):
        area.mark("inference", "blocked", "inference disabled in .env.gate (smoke mode)")
        return area.finish()
    if not args.fixture:
        area.mark("fixture", "blocked", "--fixture not given")
        return area.finish()
    fixture = Path(args.fixture)
    g.verify_fixture(fixture)
    # Installed before the model-area lock wait: a SIGTERM while this area waits for the area before it
    # ends the wait as blocked ("stopped by SIGTERM") instead of killing the process mid-way.
    g.install_stop_handler()
    try:
        with g.model_area_slot(name, lambda r: g.append_jsonl(area.dir / "load-gate.jsonl", r)):
            return _run_area(area, name, args, env, select, turns, mutate, fixture)
    except g.GateBlocked as exc:  # waited too long for another model area (or a gate refusal outside the setup)
        kind = exc.result.get("kind")
        if kind == "stopped":
            area.mark("stopped", "blocked", "stopped by SIGTERM (area timeout) while waiting for the model-area lock; nothing was loaded")
        else:
            serial = kind == "area-wait-timeout"
            area.mark("serial" if serial else "load-gate", "blocked",
                      ("model areas run one at a time: " if serial else "MS-01 load gate: ") + str(exc))
        return area.finish()


def _run_area(area, name, args, env, select, turns, mutate, fixture) -> int:
    idle_total = [0.0]
    user = g.tenant_id(args.tenant_label or f"area-{name}")
    c = g.Client(args.base_url, env, log=area.log_request)

    def log(r):
        idle_total[0] += r.get("waited_s", 0.0) if r.get("kind") == "idle-wait" else 0.0
        g.append_jsonl(area.dir / "load-gate.jsonl", r)
    try:
        loaded = g.load_fixture(c, user, fixture, env, log)
        back = g.readback(c, user, fixture, env=env, gate_log=log)
        if not area.check("fixture-loaded", not loaded["failures"] and not back["mismatched"] and not back["markers_missing"],
                          written=loaded["written"]):
            return area.finish({"tenant": user, "idle_wait_s": round(idle_total[0], 3)})
        if mutate:
            m = g.apply_mutations(c, user, fixture, env, log)
            if not area.check("mutations-applied", m["applied"] == m["total"], applied=m["applied"], total=m["total"]):
                return area.finish({"tenant": user, "idle_wait_s": round(idle_total[0], 3)})
        baseline = harness.take_baseline(c, user, env, log)
    except g.GateBlocked as exc:
        area.mark("setup", "blocked", f"stopped by SIGTERM (area timeout) during setup" if exc.result.get("kind") == "stopped"
                  else f"MS-01 load gate: {exc}")
        return area.finish({"tenant": user, "idle_wait_s": round(idle_total[0], 3)})
    if not area.check("baseline-settled", baseline["ok"], files=len(baseline["files"])):
        return area.finish({"tenant": user, "idle_wait_s": round(idle_total[0], 3)})
    rubric = json.loads((GATE / "rubric.json").read_text(encoding="utf-8"))
    index = harness.entry_index(fixture)
    rows = area.dir / "answers.jsonl"
    rows.write_text("", encoding="utf-8")
    for q in [q for q in g.read_jsonl(fixture / "truth" / "queries.jsonl") if select(q)]:
        if g.stop_requested():
            area.mark(q["id"], "blocked", "stopped by SIGTERM (area timeout); the tenant is restored")
            return area.finish({"tenant": user, "answers": str(rows), "idle_wait_s": round(idle_total[0], 3)})
        def on_row(n, text, spec, res, trace, shape, q=q):
            sig = score.answer_signals(spec, res["answer"], rubric) if res["status"] == 200 else None
            g.append_jsonl(rows, {"id": f"{name}:{q['id']}#{n}", "query": q["id"], "turn": n, "text": text, "http": res["status"],
                                  "answer": res["answer"], "signals": sig, "trace": trace if trace else shape,
                                  "idle_wait_s": res.get("idle_wait_s"), "verdict": "needs-judge"})
            if res["status"] != 200:
                area.check(f"{q['id']}#{n}-http", False, http=res["status"])
            else:
                area.mark(f"{q['id']}#{n}", "untested", "needs-judge (rubric.json)")
        stop = harness.run_turns(c, user, q, turns(q), env, log, index, baseline, on_row,
                                 lambda r: g.append_jsonl(area.dir / "restore.jsonl", r))
        if stop:
            area.mark(q["id"], "blocked", stop)
            return area.finish({"tenant": user, "answers": str(rows), "idle_wait_s": round(idle_total[0], 3)})
    return area.finish({"tenant": user, "answers": str(rows), "idle_wait_s": round(idle_total[0], 3)})
