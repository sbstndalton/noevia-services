"""Explain why a model failed to load, from the llama.cpp router's own log.

The router spawns one child per model ("spawning server instance with name=X on port P"),
prefixes the child's output with "[P]", and reports "instance name=X exited with status N".
A non-zero exit is not always a failure: unloading a model that is busy ends with the
router force-killing it, which is a requested stop. Those are recognised and ignored, a
later successful load clears an earlier failure, and only the container's current run is
read, so problems from before a restart never linger.

Diagnosis only proposes what to check. When the cause is not one it recognises it says so.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SPAWN = re.compile(r"spawning server instance with name=(\S+) on port (\d+)")
_EXIT = re.compile(r"instance name=(\S+) exited with status (-?\d+)")
_FORCED = re.compile(r"(force-killing model instance name=(\S+))|(unload: stopping model instance name=(\S+))")
_PREFIX = re.compile(r"^\[(\d+)\]\s?(.*)$")

# (id, title, patterns, advice). Order matters: the first matching rule wins.
RULES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("kv-cache", "The KV cache did not fit",
     (r"failed to allocate.*kv", r"kv.?cache.*(alloc|fail)", r"llama_kv_cache.*failed"),
     "Lower the context size or parallel slots, or quantize the cache (cache-type-k/v = q8_0)."),
    ("gpu-memory", "The GPU ran out of memory",
     (r"out of (device )?memory", r"ErrorOutOfDeviceMemory", r"Device memory allocation of size .* failed",
      r"cudaMalloc failed", r"unable to allocate .*buffer", r"failed to allocate .*buffer", r"hipMalloc.*fail"),
     "Lower the context size or GPU layers, remove the vision projector if you do not need it, "
     "or set fit = on so llama.cpp places layers itself."),
    ("missing-file", "A model file is missing",
     (r"failed to open", r"No such file or directory", r"model file not found", r"mmproj.*(not found|failed to load)"),
     "Check the model = and mmproj = paths in this model's settings; the file may have been moved or deleted."),
    ("corrupt-file", "The model file is corrupt or incomplete",
     (r"invalid magic", r"failed to read (the )?(magic|header|tensor)", r"tensor .* data is not within the file bounds",
      r"file is truncated", r"unexpected end of file"),
     "The download is probably incomplete. Delete the file and download it again."),
    ("architecture", "This llama.cpp build does not support the model",
     (r"unknown model architecture", r"unsupported model architecture", r"unknown (pre-tokenizer|tokenizer) type"),
     "Use a newer llama.cpp image, or choose a model whose architecture this build supports."),
    ("bad-option", "A setting is not valid for this llama.cpp build",
     (r"error: (invalid|unknown) argument", r"unknown argument", r"invalid value for"),
     "Remove or correct the setting named in the log; it may be misspelled or unsupported by this build."),
    ("port", "The server port is already in use",
     (r"address already in use", r"couldn't bind", r"failed to bind"),
     "Another process holds the port. Restart the llama.cpp container."),
    ("context", "The context exceeds what the model was trained for",
     (r"exceeds the (model's )?(training|trained) context", r"n_ctx_train",),
     "Reduce the context size to the model's trained length or configure RoPE scaling deliberately."),
)


@dataclass
class Diagnosis:
    model: str
    status: int
    cause: str            # rule id, or "unknown"
    title: str
    advice: str
    evidence: list[str] = field(default_factory=list)


def _match(lines: list[str]) -> tuple[str, str, str, list[str]]:
    for rule_id, title, patterns, advice in RULES:
        hits = [line for line in lines if any(re.search(p, line, re.I) for p in patterns)]
        if hits:
            return rule_id, title, advice, hits[-4:]
    tail = [line for line in lines if re.search(r"\b(error|failed|fatal|abort)", line, re.I)][-4:] or lines[-4:]
    return ("unknown", "The model stopped for a reason noevia does not recognise",
            "Read the log lines below; they are the last output before the model stopped.", tail)


def analyse(text: str) -> list[Diagnosis]:
    """Current, unresolved load failures in router log text, newest first."""
    port_of: dict[str, str] = {}          # model -> port of its latest instance
    output: dict[str, list[str]] = {}     # port -> that instance's output
    requested: set[str] = set()           # models whose current instance was asked to stop
    failures: dict[str, tuple[str, Diagnosis]] = {}  # model -> (failed port, diagnosis)
    for raw in text.splitlines():
        m = _PREFIX.match(raw)
        if m:
            output.setdefault(m.group(1), []).append(m.group(2))
            continue
        spawn = _SPAWN.search(raw)
        if spawn:
            name, port = spawn.group(1), spawn.group(2)
            port_of[name], output[port] = port, []
            requested.discard(name)
            continue
        forced = _FORCED.search(raw)
        if forced:
            requested.add(forced.group(2) or forced.group(4))
            continue
        ended = _EXIT.search(raw)
        if not ended:
            continue
        name, status = ended.group(1), int(ended.group(2))
        port = port_of.get(name, "")
        if status == 0 or name in requested:
            requested.discard(name)
            continue
        cause, title, advice, evidence = _match(output.get(port, []))
        failures[name] = (port, Diagnosis(name, status, cause, title, advice, evidence))
    # A later instance of the same model that loaded successfully clears the failure;
    # the failed instance itself may have loaded and then crashed on its first request.
    for name in list(failures):
        failed_port = failures[name][0]
        latest = port_of.get(name, "")
        if latest and latest != failed_port and any("model loaded" in line for line in output.get(latest, [])):
            failures.pop(name)
    return [d for _, d in reversed(list(failures.values()))]
