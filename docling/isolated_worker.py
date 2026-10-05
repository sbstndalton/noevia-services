"""Child half of isolation.py: a long-lived process that runs conversions on request.

Protocol, deliberately tiny so it cannot be confused by anything Docling prints:
  * requests arrive on stdin, one JSON object per line: {"path": ..., "name": ...}
  * replies go to a dedicated pipe (the fd given on the command line), each as a
    4-byte big-endian length followed by that many bytes of JSON:
        {"ok": <result>}   the target returned
        {"error": true}    the target raised; no message, because exception text
                           can quote document content and must not travel anywhere
  * once the target module is imported the child sends one {"ready": true} frame, so the
    parent can tell "could not start" (spawn/import failure: infrastructure) from "the
    conversion failed" (the document); the models themselves load lazily, on first use
  * stdout is the null device, so a stray print from a library cannot corrupt a reply.

Usage: python isolated_worker.py <reply-fd> <module:function>
"""
import importlib
import json
import os
import struct
import sys


def main(argv):
    reply_fd, target = int(argv[1]), argv[2]
    module_name, function_name = target.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    # Buffered, and flushed after every frame: BufferedWriter loops until the whole frame is
    # written, whereas an unbuffered write to a pipe may be partial and truncate it.
    reply = os.fdopen(reply_fd, "wb")

    def send(payload):
        data = json.dumps(payload).encode()
        reply.write(struct.pack(">I", len(data)) + data)
        reply.flush()

    send({"ready": True})
    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            payload = {"ok": function(request["path"], request["name"])}
        except BaseException:
            payload = {"error": True}
        try:
            send(payload)
        except (TypeError, ValueError):
            send({"error": True})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
