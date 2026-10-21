#!/usr/bin/env python3
"""Run the ten workflow areas in the order the gate needs.

    run_areas.py --run-dir RUN --fixture FIX [--env-file F] [--base-url URL] [--only a,b] [--parallel-model-areas]

Order (one stage after the other; the areas inside a stage run together):

  1. capture, tenancy            no model; each has its own tenants on the no-model diary, so they run in parallel
  2. semantic, temporal, corrections, follow_ups, confirmed_vs_inferred
                                 model areas: ONE AT A TIME. Four of them in parallel starve each other on the
                                 shared inference lock and their queries time out (services#113).
  3. error_paths, restart, budgets   alone, one at a time, nothing else running

--parallel-model-areas puts the model areas in one stage and sets GATE_MODEL_AREAS_PARALLEL=1 for them (opt in).
Without it the children run with that variable removed, so even a model area started on its own elsewhere
waits for the area before it (gatelib.model_area_slot). An area that is blocked does not stop the others:
every area prints its own result line, and the order actually run (with start and end times) is written
to RUN/workflows/areas-run.json. Exit status 0 when every area exits 0.

Standard library only; run with `python3 -I`. The runner makes no request itself.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gatelib as g  # noqa: E402

NO_MODEL_PARALLEL = ("capture", "tenancy")
MODEL_AREAS = ("semantic", "temporal", "corrections", "follow_ups", "confirmed_vs_inferred")
ALONE = ("error_paths", "restart", "budgets")
NEEDS_FIXTURE = frozenset(MODEL_AREAS) | {"budgets"}
AREA_TIMEOUT_S = float(os.environ.get("GATE_AREA_TIMEOUT_S", "10800"))
# After the timeout the area gets SIGTERM and this long to stop (a model area then finishes its restore
# and writes its result); only after that is it killed.
AREA_GRACE_S = float(os.environ.get("GATE_AREA_GRACE_S", "60"))


def plan(parallel_model: bool = False, only=None) -> list:
    """Stages, each a list of areas that may run at the same time."""
    stages = [list(NO_MODEL_PARALLEL)]
    stages += [list(MODEL_AREAS)] if parallel_model else [[a] for a in MODEL_AREAS]
    stages += [[a] for a in ALONE]
    if only:
        stages = [[a for a in st if a in only] for st in stages]
    return [st for st in stages if st]


def _command(area: str, args) -> list:
    cmd = [sys.executable, "-I", str(HERE / "areas" / f"{area}.py"), "--run-dir", args.run_dir, "--env-file", args.env_file]
    if args.fixture and area in NEEDS_FIXTURE:
        cmd += ["--fixture", args.fixture]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    return cmd


def subprocess_runner(area: str, args, parallel_model: bool) -> dict:
    env = dict(os.environ)
    if parallel_model:
        env["GATE_MODEL_AREAS_PARALLEL"] = "1"
    else:
        env.pop("GATE_MODEL_AREAS_PARALLEL", None)
    proc = subprocess.Popen(_command(area, args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        stdout, stderr = proc.communicate(timeout=AREA_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _stop_overdue(proc, area)
    return {"rc": proc.returncode, "out": (stdout or "").strip().splitlines()[-1] if (stdout or "").strip() else
            (stderr or "").strip()[-300:]}


def _stop_overdue(proc, area: str) -> dict:
    """Timeout: SIGTERM first with a grace period (the area can finish its restore), then SIGKILL. The
    result is blocked; when the area had to be killed the evidence says its tenant may be unrestored."""
    killed = False
    proc.terminate()
    try:
        proc.communicate(timeout=AREA_GRACE_S)
    except subprocess.TimeoutExpired:
        killed = True
        proc.kill()
        proc.communicate()
    reason = f"area still running after {AREA_TIMEOUT_S:.0f} s (GATE_AREA_TIMEOUT_S); "
    reason += (f"did not stop within {AREA_GRACE_S:.0f} s of SIGTERM (GATE_AREA_GRACE_S) and was killed: "
               "its tenant may be unrestored (a query's logged exchange may remain)") if killed else "stopped after SIGTERM"
    return {"rc": 124, "killed": killed, "out": json.dumps({"area": area, "status": "blocked", "killed": killed,
                                                            "tenant_may_be_unrestored": killed, "reason": reason})}


def run_all(args, runner=subprocess_runner, clock=time.time) -> int:
    parallel_model = bool(args.parallel_model_areas)
    only = set(args.only.split(",")) if args.only else None
    stages, order, rc = plan(parallel_model, only), [], 0
    for stage in stages:
        def one(area):
            started = clock()
            res = runner(area, args, parallel_model)
            return {"area": area, "rc": res["rc"], "started": started, "finished": clock(), "result": res.get("out", ""),
                    "killed": bool(res.get("killed")), "tenant_may_be_unrestored": bool(res.get("killed"))}
        if len(stage) == 1:
            done = [one(stage[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(stage)) as pool:
                done = list(pool.map(one, stage))
        for row in done:
            order.append(row)
            rc = rc or (1 if row["rc"] else 0)
            print(json.dumps({"area": row["area"], "rc": row["rc"], "result": row["result"]}, sort_keys=True))
    record = {"parallel_model_areas": parallel_model, "stages": stages, "order": [r["area"] for r in order], "runs": order}
    g.write_json(Path(args.run_dir) / "workflows" / "areas-run.json", record)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--env-file", default=str(g.DEFAULT_ENV))
    ap.add_argument("--base-url", default=None, help="passed to every area; default: each area's own diary")
    ap.add_argument("--only", default="", help="comma-separated area script names (capture,semantic,error_paths,...)")
    ap.add_argument("--parallel-model-areas", action="store_true",
                    help="run the model areas at the same time (they then starve each other on the inference lock)")
    return run_all(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
