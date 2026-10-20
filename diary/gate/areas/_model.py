"""Shared runner for the workflow areas that need inference (semantic, temporal, corrections,
follow-ups, confirmed-vs-inferred). Each area loads the fixture into its OWN tenant, so these areas
are parallel-safe with each other; the MS-01 load gate still runs before every inference call.

Without inference (smoke mode) they report "blocked". With inference, answer rows are recorded with
deterministic rubric signals and stay "needs-judge" (status "untested") unless a request fails.
"""
from __future__ import annotations

import json
import sys
import uuid
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
    user = g.tenant_id(args.tenant_label or f"area-{name}")
    c = g.Client(args.base_url, env, log=area.log_request)
    loaded = g.load_fixture(c, user, fixture)
    back = g.readback(c, user, fixture)
    if not area.check("fixture-loaded", not loaded["failures"] and not back["mismatched"] and not back["markers_missing"],
                      written=loaded["written"]):
        return area.finish({"tenant": user})
    if mutate:
        m = g.apply_mutations(c, user, fixture)
        if not area.check("mutations-applied", m["applied"] == m["total"], applied=m["applied"], total=m["total"]):
            return area.finish({"tenant": user})
    rubric = json.loads((GATE / "rubric.json").read_text(encoding="utf-8"))
    index = harness.entry_index(fixture)
    rows = area.dir / "answers.jsonl"
    rows.write_text("", encoding="utf-8")
    for q in [q for q in g.read_jsonl(fixture / "truth" / "queries.jsonl") if select(q)]:
        history, session = [], f"{name}-{q['id']}-{uuid.uuid4().hex[:8]}"
        for n, (text, spec) in enumerate(turns(q)):
            gate = g.load_gate(env)
            g.append_jsonl(area.dir / "load-gate.jsonl", {"query": q["id"], "turn": n, **gate})
            if not gate["ok"]:
                area.mark(f"{q['id']}#{n}", "blocked", "MS-01 load gate: " + "; ".join(gate["reasons"]))
                return area.finish({"tenant": user, "stopped": "load gate"})
            res = harness.ask(c, user, text, q["clock"], session, history)
            trace = harness.normalize_trace(res["trace"], index) if res["trace"] is not None else None
            sig = score.answer_signals(spec, res["answer"], rubric) if res["status"] == 200 else None
            g.append_jsonl(rows, {"id": q["id"], "turn": n, "text": text, "http": res["status"], "answer": res["answer"],
                                  "signals": sig, "trace": trace if trace else "trace unavailable", "verdict": "needs-judge"})
            if res["status"] != 200:
                area.check(f"{q['id']}#{n}-http", False, http=res["status"])
                break
            area.mark(f"{q['id']}#{n}", "untested", "needs-judge (rubric.json)")
            history += [{"role": "user", "content": text}, {"role": "assistant", "content": res["answer"]}]
    return area.finish({"tenant": user, "answers": str(rows)})
