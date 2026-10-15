"""Child half of isolation.py: a long-lived process that runs conversions on request.

Protocol, deliberately tiny so it cannot be confused by anything Docling prints:
  * requests arrive on stdin, one JSON object per line: {"path": ..., "name": ...}, plus
    "figures": true when the caller wants the document's figures too (#1726), and
    "pages": [first, last] when only that page range is wanted (#1864)
  * replies go to a dedicated pipe (the fd given on the command line), each as a
    4-byte big-endian length followed by that many bytes of JSON:
        {"ok": <result>}   the target returned
        {"error": true}    the target raised; no message, because exception text
                           can quote document content and must not travel anywhere
  * once the target module is imported the child sends one {"ready": true} frame, so the
    parent can tell "could not start" (spawn/import failure: infrastructure) from "the
    conversion failed" (the document); the models themselves load lazily, on first use
  * stdout is the null device, so a stray print from a library cannot corrupt a reply.

Usage: python isolated_worker.py <reply-fd>|--stdout-replies <module:function>
"""
import importlib
import json
import os
import struct
import sys


def main(argv):
    target = argv[2]
    if argv[1] == "--stdout-replies":
        # The native parent cannot pass arbitrary inherited descriptors without unsafe
        # platform glue. Preserve the original stdout pipe for framing, then replace
        # fd 1 itself before importing the engine: C and Python prints both go nowhere.
        reply_fd = os.dup(1)
        with open(os.devnull, "wb") as null:
            os.dup2(null.fileno(), 1)
    else:
        reply_fd = int(argv[1])
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
            extra = {"figures": True} if request.get("figures") is True else {}
            # #1864: a page range, two positive integers, first <= last.
            pages = request.get("pages")
            if isinstance(pages, list) and len(pages) == 2 and all(type(n) is int and n >= 1 for n in pages) and pages[0] <= pages[1]:
                extra["pages"] = (pages[0], pages[1])
            payload = {"ok": function(request["path"], request["name"], **extra)}
        except BaseException:
            payload = {"error": True}
        try:
            send(payload)
        except (TypeError, ValueError):
            send({"error": True})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
