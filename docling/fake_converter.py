"""A stand-in conversion target for the server and isolation tests.

Runs inside the child process (`IsolatedExtractor("fake_converter:extract")`), so
the real kill/timeout/disconnect machinery is exercised without Docling. The
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


def extract(path, name):
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
    return {"pages": [{"number": 1, "text": "ok", "status": "native",
                       "method": "docling", "truncated": False}],
            "total": 1, "truncatedPages": False, "pid": os.getpid()}


def big(path, name):
    return {"pages": [{"number": 1, "text": "x" * 190_000, "status": "native",
                       "method": "docling", "truncated": False}],
            "total": 1, "truncatedPages": False}
