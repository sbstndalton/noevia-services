"""Bounded scores-only transport for the optional Diary score-gap Rust leaf.

No content, IDs, ranks or tenant state enter the child. The installed binary is
fixed; image pairing and selector activation require a later release change.
At most four transports run concurrently. Each child has a two-second wait and
bounded cleanup joins; failures kill its private process group and reap it.
The host retains all candidates on unavailable rather than applying Python cut.
Protocol reference: noevia-rs CLI 0dce0eaa101ee751d23280b0eb6da34c285b266f
(stacked on foundation 41f50e63d62f123289d85291b368c8e86770f840), #1071.
Neither a release pin nor an environment selector is activated by this adapter.
"""
from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import threading

BINARY = "/usr/local/bin/diary-score-gap"
TIMEOUT = 2.0
MAX_INPUT = 131072
MAX_OUTPUT = 32768
MAX_SCORES = 4096
_SLOTS = threading.BoundedSemaphore(4)
_VERSION = "diary-score-gap-v1"
_NUMBER = re.compile(r"[+\-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+\-]?[0-9]+)?\Z")
_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _number(token):
    if token == "null":
        return None
    if not _NUMBER.fullmatch(token):
        raise ValueError("number")
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("finite")
    return value


def _indices(field, size):
    if not field:
        return []
    tokens = field.split(",")
    if len(tokens) > size or any(not _INDEX.fullmatch(t) or len(t) > 4 for t in tokens):
        raise ValueError("positions")
    values = [int(t) for t in tokens]
    if any(a >= b for a, b in zip(values, values[1:])) or any(v >= size for v in values):
        raise ValueError("order")
    return values


def _decode(body, size, expected_best):
    text = body.decode("ascii")
    if not text.endswith("\n"):
        raise ValueError("framing")
    fields = text[:-1].split("\t")
    if len(fields) != 4 or fields[0] != _VERSION:
        raise ValueError("version")
    best = _number(fields[1])
    if best != expected_best or (best is not None and best == 0 and
                                math.copysign(1, best) != math.copysign(1, expected_best)):
        raise ValueError("best")
    kept, removed = _indices(fields[2], size), _indices(fields[3], size)
    if sorted(kept + removed) != list(range(size)):
        raise ValueError("partition")
    return kept, removed, best


def partition(scores, gap, best):
    """Return verified positions/best, or None: caller keeps all on unavailable.

    Scores have already undergone Python coercion/nonfinite normalization. No
    filtering is recomputed here; only transport and response integrity checks.
    """
    if not _SLOTS.acquire(blocking=False):
        return None
    try:
        return _partition(scores, gap, best)
    finally:
        _SLOTS.release()


def _partition(scores, gap, best):
    if len(scores) > MAX_SCORES:
        return None
    token = lambda v: "null" if v is None else repr(v)
    payload = f"{_VERSION}\t{token(gap)}\t{','.join(map(token, scores))}\n".encode("ascii")
    if len(payload) > MAX_INPUT:
        return None
    try:
        process = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env={"LANG": "C", "LC_ALL": "C"},
                                   close_fds=True, start_new_session=True)
    except (OSError, ValueError):
        return None
    chunks = {}
    failed = threading.Event()

    def kill():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass

    def read(name, stream, limit):
        try:
            data = stream.read(limit + 1)
            chunks[name] = data
            if len(data) > limit:
                failed.set()
                kill()
        except (OSError, ValueError):
            failed.set()
            kill()
        finally:
            stream.close()

    def write():
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (OSError, ValueError):
            failed.set()
            kill()
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass

    workers = [threading.Thread(target=write, daemon=True),
               threading.Thread(target=read, args=("out", process.stdout, MAX_OUTPUT), daemon=True),
               threading.Thread(target=read, args=("err", process.stderr, 128), daemon=True)]
    try:
        for worker in workers:
            worker.start()
        process.wait(timeout=TIMEOUT)
    except (subprocess.TimeoutExpired, OSError, RuntimeError):
        failed.set()
        kill()
    finally:
        # A killed direct child is always reaped. The trusted installed binary
        # never launches descendants; reader/writer joins are bounded as defense.
        try:
            process.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            failed.set()
            kill()
            process.wait()
        except OSError:
            failed.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=TIMEOUT)
                if worker.is_alive():
                    failed.set()
                    kill()
                    worker.join(timeout=TIMEOUT)
        if not any(worker.is_alive() for worker in workers):
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    failed.set()
    if failed.is_set() or process.returncode != 0 or chunks.get("err") != b"":
        return None
    try:
        result = _decode(chunks.get("out", b""), len(scores), best)
        # Disabled/name-suppressed gaps and all-unknown lists cannot remove rows.
        # This is caller policy validation, not a second score-gap calculation.
        if result[1] and (gap is None or best is None):
            return None
        return result
    except (ValueError, UnicodeError, OverflowError):
        return None
