"""Opt-in, bounded source-only corpus write transform subprocess.

Python owns storage, journal order, consent, XID, and locale. No fallback after
DIARY_CORPUS_WRITE_IMPL=rust is selected. The binary path is fixed at packaging.
"""
from __future__ import annotations

import json
import os
import selectors
import subprocess
import time

BINARY = "/usr/local/bin/diary-corpus-write-cli"
SELECTOR = "DIARY_CORPUS_WRITE_IMPL"
INPUT_MAGIC = b"DCW1"
OUTPUT_MAGIC = b"DWR1"
END = b"END1"
MAX_INPUT = 64 * 1024 * 1024
MAX_OUTPUT = 128 * 1024 * 1024
MAX_STDERR = 256
TIMEOUT_S = 15.0


class NativeWriteUnavailable(OSError):
    """The selected native transform could not safely return a result."""


def implementation() -> str:
    value = (os.environ.get(SELECTOR) or "python").strip().lower()
    if value not in ("python", "rust"):
        raise ValueError(f"{SELECTOR} must be python or rust")
    return value


def _expect(response: object, op: str, fields: dict[str, type]) -> dict:
    if not isinstance(response, dict) or set(response) != {"version", "op", *fields}:
        raise NativeWriteUnavailable("native corpus write response shape")
    if type(response["version"]) is not int or response["version"] != 1 or response["op"] != op:
        raise NativeWriteUnavailable("native corpus write response identity")
    for key, expected in fields.items():
        value = response[key]
        if expected is str or expected is bool:
            if type(value) is not expected:
                raise NativeWriteUnavailable("native corpus write response field")
        elif expected is object and value is not None and type(value) is not str:
            raise NativeWriteUnavailable("native corpus write response nullable field")
    return response


def transform(op: str, **fields) -> dict:
    schemas = {
        "prepare_append": {"time_token", "topic", "me", "assistant", "xid"},
        "apply_append": {"current", "day_header", "sub_header", "body", "xid"},
        "inspect_edit": {"current", "xid", "new_me", "new_assistant"},
        "apply_edit": {"current", "xid", "new_me", "new_assistant"},
    }
    if op not in schemas or set(fields) != schemas[op]:
        raise TypeError("invalid corpus write request shape")
    for name, value in fields.items():
        if name == "current" and value is None:
            continue
        if type(value) is not str:
            raise TypeError("invalid corpus write request field")
    if not os.path.isfile(BINARY) or not os.access(BINARY, os.X_OK):
        raise NativeWriteUnavailable("native corpus write binary unavailable")
    data = json.dumps({"version": 1, "op": op, **fields}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(data) + 16 > MAX_INPUT:
        raise NativeWriteUnavailable("native corpus write input too large")
    frame = INPUT_MAGIC + len(data).to_bytes(8, "big") + data + END
    deadline = time.monotonic() + TIMEOUT_S
    try:
        proc = subprocess.Popen([BINARY, "transform-v1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0, close_fds=True,
                                env={"PATH": "/usr/bin:/bin"})
    except OSError as exc:
        raise NativeWriteUnavailable("native corpus write spawn failed") from exc
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    sent = 0
    try:
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            os.set_blocking(stream.fileno(), False)
        selector.register(proc.stdin, selectors.EVENT_WRITE, "input")
        selector.register(proc.stdout, selectors.EVENT_READ, "output")
        selector.register(proc.stderr, selectors.EVENT_READ, "error")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeWriteUnavailable("native corpus write timeout")
            for key, _ in selector.select(remaining):
                stream = key.fileobj
                if key.data == "input":
                    try:
                        sent += os.write(stream.fileno(), frame[sent:sent + 65536])
                    except BrokenPipeError:
                        sent = len(frame)
                    if sent >= len(frame):
                        selector.unregister(stream)
                        stream.close()
                else:
                    try:
                        chunk = os.read(stream.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                    elif key.data == "output":
                        if len(stdout) + len(chunk) > MAX_OUTPUT:
                            raise NativeWriteUnavailable("native corpus write output too large")
                        stdout.extend(chunk)
                    else:
                        if len(stderr) + len(chunk) > MAX_STDERR:
                            raise NativeWriteUnavailable("native corpus write stderr too large")
                        stderr.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or proc.wait(timeout=remaining) != 0:
            raise NativeWriteUnavailable("native corpus write exit failure")
        if len(stdout) < 16 or stdout[:4] != OUTPUT_MAGIC or stdout[-4:] != END:
            raise NativeWriteUnavailable("native corpus write frame")
        length = int.from_bytes(stdout[4:12], "big")
        if length != len(stdout) - 16:
            raise NativeWriteUnavailable("native corpus write length")
        try:
            response = json.loads(stdout[12:-4])
        except (UnicodeError, ValueError) as exc:
            raise NativeWriteUnavailable("native corpus write json") from exc
        schemas = {
            "prepare_append": {"sub_header": str, "body": str},
            "apply_append": {"changed": bool, "text": str},
            "inspect_edit": {"found": bool, "current_text": object, "current_hash": object,
                             "edited_text": object, "edited_hash": object},
            "apply_edit": {"found": bool, "changed": bool, "text": object},
        }
        checked = _expect(response, op, schemas[op])
        if op == "inspect_edit":
            names = ("current_text", "current_hash", "edited_text", "edited_hash")
            if checked["found"] != all(checked[name] is not None for name in names):
                raise NativeWriteUnavailable("native corpus write edit consistency")
        elif op == "apply_edit":
            if checked["found"] != (checked["text"] is not None) or (not checked["found"] and checked["changed"]):
                raise NativeWriteUnavailable("native corpus write edit consistency")
        elif op == "apply_append" and not checked["changed"] and checked["text"] != fields["current"]:
            raise NativeWriteUnavailable("native corpus write append consistency")
        return checked
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        if isinstance(exc, NativeWriteUnavailable):
            raise
        raise NativeWriteUnavailable("native corpus write transport failure") from exc
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
