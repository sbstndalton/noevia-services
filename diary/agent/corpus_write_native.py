"""Opt-in, bounded source-only corpus write transform subprocess.

Python owns storage, journal order, consent, XID, and locale. No fallback after
DIARY_CORPUS_WRITE_IMPL=rust is selected. The binary path is fixed at packaging.
"""
from __future__ import annotations

import json
import os
import re
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
    try:
        for value in response.values():
            if isinstance(value, str):
                value.encode("utf-8")
    except UnicodeError as exc:
        raise NativeWriteUnavailable("native corpus write response encoding") from exc
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
    try:
        data = json.dumps({"version": 1, "op": op, **fields}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except UnicodeError as exc:
        raise NativeWriteUnavailable("native corpus write request encoding") from exc
    if len(data) + 16 > MAX_INPUT:
        raise NativeWriteUnavailable("native corpus write input too large")
    frame = INPUT_MAGIC + len(data).to_bytes(8, "big") + data + END
    stdout = _exchange(frame)
    if len(stdout) < 16 or stdout[:4] != OUTPUT_MAGIC or stdout[-4:] != END:
        raise NativeWriteUnavailable("native corpus write frame")
    length = int.from_bytes(stdout[4:12], "big")
    if length != len(stdout) - 16:
        raise NativeWriteUnavailable("native corpus write length")
    try:
        response = json.loads(stdout[12:-4].decode("utf-8"),
                              object_pairs_hook=_unique_keys,
                              parse_constant=_invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
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
        if not all((checked[name] is not None) == checked["found"] for name in names):
            raise NativeWriteUnavailable("native corpus write edit consistency")
        if checked["found"] and any(not re.fullmatch(r"[0-9a-f]{64}", checked[name])
                                    for name in ("current_hash", "edited_hash")):
            raise NativeWriteUnavailable("native corpus write edit hash")
    elif op == "apply_edit":
        if checked["found"] != (checked["text"] is not None) or (not checked["found"] and checked["changed"]):
            raise NativeWriteUnavailable("native corpus write edit consistency")
    elif op == "apply_append" and not checked["changed"] and checked["text"] != fields["current"]:
        raise NativeWriteUnavailable("native corpus write append consistency")
    return checked


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate response key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("invalid JSON constant")


def _exchange(frame):
    """Trust a reply only after complete input, EOF, clean exit and cleanup."""
    deadline = time.monotonic() + TIMEOUT_S
    selector = None
    proc = None
    code = None
    fault = None
    sent = 0
    fed = False
    feed_error = False
    stdout = bytearray()
    stderr = bytearray()

    def close(stream):
        selector.unregister(stream)
        stream.close()

    try:
        # Construct before spawn: failure cannot strand a child. All subsequent
        # setup and cancellation are covered by the same ownership finally.
        selector = selectors.DefaultSelector()
        proc = subprocess.Popen([BINARY, "transform-v1"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=0, close_fds=True, env={"PATH": "/usr/bin:/bin"})
        for stream, event, kind in ((proc.stdin, selectors.EVENT_WRITE, "input"),
                                     (proc.stdout, selectors.EVENT_READ, "output"),
                                     (proc.stderr, selectors.EVENT_READ, "error")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, event, kind)
        while selector.get_map() or proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeWriteUnavailable("native corpus write timeout")
            for key, _ in selector.select(min(.05, remaining)):
                stream = key.fileobj
                try:
                    if key.data == "input":
                        if sent == len(frame):
                            close(stream)
                            fed = True
                            continue
                        count = os.write(stream.fileno(), memoryview(frame)[sent:sent + 65536])
                        if count == 0:
                            raise OSError("short input write")
                        sent += count
                    else:
                        output = stdout if key.data == "output" else stderr
                        cap = MAX_OUTPUT if key.data == "output" else MAX_STDERR
                        chunk = os.read(stream.fileno(), min(65536, cap + 1 - len(output)))
                        if not chunk:
                            close(stream)
                        else:
                            output.extend(chunk)
                            if len(output) > cap:
                                raise NativeWriteUnavailable("native corpus write stream too large")
                except BlockingIOError:
                    continue
                except BrokenPipeError:
                    if key.data != "input":
                        raise
                    feed_error = True
                    close(stream)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        fault = "transport"
    finally:
        # Nested cleanup ensures selector/pipe close failures cannot bypass reap.
        # BaseException cancellation propagates only after this owned cleanup.
        try:
            if proc is not None:
                try:
                    if proc.poll() is None:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                finally:
                    code = proc.wait()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            fault = fault or "cleanup"
        finally:
            try:
                if proc is not None:
                    for stream in (proc.stdin, proc.stdout, proc.stderr):
                        try:
                            stream.close()
                        except (OSError, ValueError, RuntimeError):
                            fault = fault or "cleanup"
            finally:
                if selector is not None:
                    try:
                        selector.close()
                    except (OSError, ValueError, RuntimeError):
                        fault = fault or "cleanup"
    if time.monotonic() >= deadline:
        fault = fault or "timeout"
    if fault or code != 0 or not fed or feed_error or stderr:
        # Never include a child's potentially private output in the exception.
        raise NativeWriteUnavailable("native corpus write unavailable")
    return stdout
