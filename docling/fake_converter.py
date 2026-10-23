"""A stand-in conversion target for the native image contract tests.

Runs inside the worker child process (isolated_worker.py, started by the native
server), so the real kill/timeout/disconnect machinery is exercised without Docling. The
document body is a directive:

    SLEEP <seconds>     block, like a conversion that never finishes
    BOOM                raise with a canary in the message
    EXIT                die at once, like an OOM kill
    FORK <pid-file>     start a grandchild `sleep`, record its pid, then block
    (anything else)     return a one-page result
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SECRET = "PATIENT NAME: REDACTED-CANARY"


def extract(path, name, figures=False, pages=None):
    directive = Path(path).read_bytes().decode("latin-1").split()
    word = directive[0] if directive else ""
    if word == "SLEEP":
        time.sleep(float(directive[1]))
    elif word == "BOOM":
        raise RuntimeError(SECRET)
    elif word == "EXIT":
        os._exit(137)
    elif word == "FORK":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        Path(directive[1]).write_text(str(child.pid))
        time.sleep(120)
    out = {"pages": [{"number": 1, "text": "ok", "status": "native",
                      "method": "docling", "truncated": False}],
           "total": 1, "truncatedPages": False, "pid": os.getpid()}
    if figures:
        out["figures"] = []  # #1726: the caller asked for figures; this stand-in has none
    if pages:
        # #1864: a range is echoed back, one page per number, out of a 1000-page document.
        out["pages"] = [{"number": n, "text": f"page {n}", "status": "native", "method": "docling", "truncated": False}
                        for n in range(pages[0], pages[1] + 1)]
        out["total"], out["truncatedPages"] = 1000, pages[1] < 1000
    return out


def big(path, name):
    return {"pages": [{"number": 1, "text": "x" * 190_000, "status": "native",
                       "method": "docling", "truncated": False}],
            "total": 1, "truncatedPages": False}
