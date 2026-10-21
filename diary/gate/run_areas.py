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

Stopping. Every area runs in its own session, so Ctrl-C at a terminal or a signal to the process group of
this runner cannot reach it behind the back of the runner. An area that overruns GATE_AREA_TIMEOUT_S, and
every area still running when this runner gets SIGTERM or SIGINT, is sent SIGTERM (the stop request: a
model area finishes its current call, restores its tenant and stops as blocked), is given GATE_AREA_GRACE_S
to do that, and only then is it killed. The last result line and the return code of the child are kept in
areas-run.json next to `timed_out: true` (or `interrupted: true`); `tenant_may_be_unrestored` is true only
for a kill. Such an area is blocked, never a pass. This runner never installs the "finish the current
query" handler of the areas on itself: it forwards the stop to its children and exits with status 3
(blocked). Whatever else ends the runner (an exception, a closed terminal), it terminates every child it
started first.

Standard library only; run with `python3 -I`. The runner makes no request itself.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gatelib as g  # noqa: E402

NO_MODEL_PARALLEL = ("capture", "tenancy")
MODEL_AREAS = ("semantic", "temporal", "corrections", "follow_ups", "confirmed_vs_inferred")
ALONE = ("error_paths", "restart", "budgets")
NEEDS_FIXTURE = frozenset(MODEL_AREAS) | {"budgets"}
AREA_TIMEOUT_S = float(os.environ.get("GATE_AREA_TIMEOUT_S", "10800"))
# After the timeout (or a stop of this runner) the area gets SIGTERM and this long to stop: a model area
# finishes the chat call it is on (up to the diary chat timeout) and then restores its tenant (idle wait,
# writes, settle), and only after that is it killed. The default is that chat timeout plus RESTORE_ALLOWANCE_S
# (1200 s); GATE_AREA_GRACE_S overrides it.
RESTORE_ALLOWANCE_S = 600.0
AREA_GRACE_S = float(os.environ.get("GATE_AREA_GRACE_S", g.CHAT_TIMEOUT_S + RESTORE_ALLOWANCE_S))
POLL_S = 1.0  # how often a waiting runner looks at the overdue deadline and at a stop request
BLOCKED_EXIT = 3  # the runner stopped before every area finished (SIGTERM/SIGINT); same code as harness.py blocked

_clock = time.monotonic  # replaced by fake clocks in the tests
_INTERRUPT = threading.Event()  # set once this runner itself must stop
_INTERRUPT_SIGNAL = [None]
_ACTIVE: dict = {}  # id(Popen) -> running Popen
_ACTIVE_LOCK = threading.RLock()


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


def _command(area: str, args) -> list:
    cmd = [sys.executable, "-I", str(HERE / "areas" / f"{area}.py"), "--run-dir", args.run_dir, "--env-file", args.env_file]
    if args.fixture and area in NEEDS_FIXTURE:
        cmd += ["--fixture", args.fixture]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    return cmd


def _last_line(stdout, stderr) -> str:
    return (stdout or "").strip().splitlines()[-1] if (stdout or "").strip() else (stderr or "").strip()[-300:]


def _forward_stop() -> None:
    """SIGTERM (the stop request) to every area child that is running."""
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE.values())
    for proc in procs:
        try:
            proc.terminate()
        except OSError:
            pass


def _on_signal(signum, frame):  # noqa: ARG001
    """Parent SIGTERM/SIGINT: remember it and forward the stop request to the children at once; the threads
    that wait on them then enforce the grace period. Deliberately NOT gatelib.install_stop_handler."""
    _INTERRUPT_SIGNAL[0] = _INTERRUPT_SIGNAL[0] or signum
    _INTERRUPT.set()
    _forward_stop()


@contextmanager
def _parent_signals():
    """SIGTERM and SIGINT stop the runner cleanly while the block runs; the old handlers come back after."""
    old = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old[sig] = signal.signal(sig, _on_signal)
    except ValueError:  # not the main thread: the finally clauses still reap the children
        pass
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _kill(proc) -> None:
    """SIGKILL the area and everything it started (it leads its own session, so its process group is its own).
    The group is signalled only after checking that it is not ours."""
    pid = getattr(proc, "pid", None)
    try:
        if pid and hasattr(os, "killpg") and os.getpgid(pid) == pid and pid != os.getpgrp():
            os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.kill()
    except OSError:
        pass



def _stop_child(proc, area: str, cause: str) -> dict:
    """SIGTERM first with a grace period (the area can finish its restore), then SIGKILL. The area is blocked
    whatever it printed. cause is "timeout" (the area overran) or "interrupt" (this runner was told to stop).
    A child that exits within the grace period keeps its own last stdout line and its own return code, and
    tenant_may_be_unrestored stays false; only a kill sets it (the synthetic line and rc 124 then stand in)."""
    killed, stdout, stderr = False, "", ""
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        stdout, stderr = proc.communicate(timeout=AREA_GRACE_S)
    except subprocess.TimeoutExpired:
        killed = True
        _kill(proc)
        stdout, stderr = proc.communicate()
    timed_out = cause == "timeout"
    why = (f"area still running after {AREA_TIMEOUT_S:.0f} s (GATE_AREA_TIMEOUT_S)" if timed_out
           else "the runner got a stop request (SIGTERM/SIGINT)")
    if killed:
        reason = (f"{why}; did not stop within {AREA_GRACE_S:.0f} s of SIGTERM (GATE_AREA_GRACE_S) and was killed: "
                  "its tenant may be unrestored (a logged query exchange may remain)")
        line, rc = None, 124
    else:
        line, rc = (_last_line(stdout, stderr) if (stdout or "").strip() else None), proc.returncode
        reason = f"{why}; stopped after SIGTERM" + ("" if line else " (it printed no result line)")
    if not line:
        line = json.dumps({"area": area, "status": "blocked", "killed": killed, "tenant_may_be_unrestored": killed,
                           "timed_out": timed_out, "interrupted": not timed_out, "reason": reason})
    return {"rc": rc, "killed": killed, "timed_out": timed_out, "interrupted": not timed_out, "reason": reason, "out": line}


def subprocess_runner(area: str, args, parallel_model: bool) -> dict:
    env = dict(os.environ)
    if parallel_model:
        env["GATE_MODEL_AREAS_PARALLEL"] = "1"
    else:
        env.pop("GATE_MODEL_AREAS_PARALLEL", None)
    # Its own session: a signal sent to the process group of this runner does not reach the area behind the
    # back of the runner, and a kill can take the whole process group of the area.
    proc = subprocess.Popen(_command(area, args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                            start_new_session=True)
    with _ACTIVE_LOCK:
        _ACTIVE[id(proc)] = proc
    try:
        deadline = _clock() + AREA_TIMEOUT_S
        while True:
            if _INTERRUPT.is_set():  # also covers a signal that arrived before the child was registered
                return _stop_child(proc, area, "interrupt")
            remaining = deadline - _clock()
            if remaining <= 0:
                return _stop_child(proc, area, "timeout")
            try:
                stdout, stderr = proc.communicate(timeout=min(POLL_S, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        # Exited by itself (or on the SIGTERM that the signal handler forwarded): blocked if the runner was stopping.
        return {"rc": proc.returncode, "out": _last_line(stdout, stderr), "interrupted": _INTERRUPT.is_set()}
    finally:
        # Whatever ended this function (a result, a stop, KeyboardInterrupt, any exception): no child survives.
        with _ACTIVE_LOCK:
            _ACTIVE.pop(id(proc), None)
        if _still_running(proc):
            _reap(proc)


def _still_running(proc) -> bool:
    if getattr(proc, "returncode", None) is not None:
        return False
    poll = getattr(proc, "poll", None)
    return poll() is None if poll else True


def _reap(proc) -> None:
    """SIGTERM, the grace period, then SIGKILL; for a child still running when the runner leaves."""
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.communicate(timeout=AREA_GRACE_S)
    except subprocess.TimeoutExpired:
        _kill(proc)
        proc.communicate()
    except OSError:
        pass


def _run_stage(stage, one) -> list:
    if len(stage) == 1:
        return [one(stage[0])]
    with ThreadPoolExecutor(max_workers=len(stage)) as pool:
        try:
            return list(pool.map(one, stage))
        except BaseException:
            # The pool exit waits for the area threads: tell them (and their children) to stop first.
            _INTERRUPT.set()
            _forward_stop()
            raise


def _signal_name(signum) -> str:
    try:
        return signal.Signals(signum).name
    except (ValueError, TypeError):
        return "error"  # no signal: the runner is leaving because of an exception


def run_all(args, runner=subprocess_runner, clock=time.time) -> int:
    parallel_model = bool(args.parallel_model_areas)
    only = set(args.only.split(",")) if args.only else None
    stages, order, rc, not_run = plan(parallel_model, only), [], 0, []
    _INTERRUPT.clear()
    _INTERRUPT_SIGNAL[0] = None
    try:
        with _parent_signals():
            for stage in stages:
                if _INTERRUPT.is_set():
                    not_run += stage
                    continue

                def one(area):
                    if _INTERRUPT.is_set():
                        not_run.append(area)
                        return None
                    started = clock()
                    res = runner(area, args, parallel_model)
                    return {"area": area, "rc": res["rc"], "started": started, "finished": clock(), "result": res.get("out", ""),
                            "killed": bool(res.get("killed")), "timed_out": bool(res.get("timed_out")),
                            "interrupted": bool(res.get("interrupted")),
                            "tenant_may_be_unrestored": bool(res.get("killed"))}
                for row in _run_stage(stage, one):
                    if row is None:
                        continue
                    order.append(row)
                    # An area that was stopped is blocked even when it exited 0 in the meantime.
                    rc = rc or (1 if row["rc"] or row["timed_out"] or row["interrupted"] else 0)
                    print(json.dumps({"area": row["area"], "rc": row["rc"], "timed_out": row["timed_out"], "result": row["result"]},
                                     sort_keys=True))
    finally:
        # Last line of defence for "no orphans": nothing the runner started outlives it.
        _forward_stop()
        interrupted, signame = _INTERRUPT.is_set(), _signal_name(_INTERRUPT_SIGNAL[0])
        _INTERRUPT.clear()
        _INTERRUPT_SIGNAL[0] = None
        record = {"parallel_model_areas": parallel_model, "stages": stages, "order": [r["area"] for r in order], "runs": order}
        if interrupted:
            record["interrupted"] = {"signal": signame, "not_run": not_run}
        g.write_json(Path(args.run_dir) / "workflows" / "areas-run.json", record)
    if interrupted:
        print(json.dumps({"runner": "run_areas", "status": "blocked", "reason": f"stopped by {signame}", "not_run": not_run},
                         sort_keys=True))
        return BLOCKED_EXIT
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
