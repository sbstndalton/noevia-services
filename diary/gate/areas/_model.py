"""Shared runner for the workflow areas that need inference (semantic, temporal, corrections,
follow-ups, confirmed-vs-inferred). Each area loads the verified fixture into its OWN tenant, so these
areas are parallel-safe with each other. Every write batch and every inference call runs inside
gatelib.inference_slot (cross-process lock, bounded re-checks, fail closed), and after each query the
tenant is restored to its post-mutation baseline exactly as harness.py does.

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
    user = g.tenant_id(args.tenant_label or f"area-{name}")
    c = g.Client(args.base_url, env, log=area.log_request)
    log = lambda r: g.append_jsonl(area.dir / "load-gate.jsonl", r)  # noqa: E731
    try:
        loaded = g.load_fixture(c, user, fixture, env, log)
        back = g.readback(c, user, fixture, env=env, gate_log=log)
        if not area.check("fixture-loaded", not loaded["failures"] and not back["mismatched"] and not back["markers_missing"],
                          written=loaded["written"]):
            return area.finish({"tenant": user})
        if mutate:
            m = g.apply_mutations(c, user, fixture, env, log)
            if not area.check("mutations-applied", m["applied"] == m["total"], applied=m["applied"], total=m["total"]):
                return area.finish({"tenant": user})
        baseline = harness.take_baseline(c, user, env, log)
    except g.GateBlocked as exc:
        area.mark("setup", "blocked", f"MS-01 load gate: {exc}")
        return area.finish({"tenant": user})
    if not area.check("baseline-settled", baseline["ok"], files=len(baseline["files"])):
        return area.finish({"tenant": user})
    rubric = json.loads((GATE / "rubric.json").read_text(encoding="utf-8"))
    index = harness.entry_index(fixture)
    rows = area.dir / "answers.jsonl"
    rows.write_text("", encoding="utf-8")
    for q in [q for q in g.read_jsonl(fixture / "truth" / "queries.jsonl") if select(q)]:
        def on_row(n, text, spec, res, trace, shape, q=q):
            sig = score.answer_signals(spec, res["answer"], rubric) if res["status"] == 200 else None
            g.append_jsonl(rows, {"id": f"{name}:{q['id']}#{n}", "query": q["id"], "turn": n, "text": text, "http": res["status"],
                                  "answer": res["answer"], "signals": sig, "trace": trace if trace else shape,
                                  "verdict": "needs-judge"})
            if res["status"] != 200:
                area.check(f"{q['id']}#{n}-http", False, http=res["status"])
            else:
                area.mark(f"{q['id']}#{n}", "untested", "needs-judge (rubric.json)")
        stop = harness.run_turns(c, user, q, turns(q), env, log, index, baseline, on_row,
                                 lambda r: g.append_jsonl(area.dir / "restore.jsonl", r))
        if stop:
            area.mark(q["id"], "blocked", stop)
            return area.finish({"tenant": user, "answers": str(rows)})
    return area.finish({"tenant": user, "answers": str(rows)})
