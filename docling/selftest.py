"""Verify the Docling integration against a real document, on real hardware.

This exists because extract.py was written against Docling's documented API but
never executed against a real install during development — the build
environment could not reach HuggingFace for the models. Everything else in this
change is unit-tested with Docling stubbed out; this is the one thing that
cannot be, so run it once on the host before trusting the sidecar:

    docker compose -f compose.yaml -f compose.docling.yaml run --rm --entrypoint python \
        docling selftest.py /path/to/sample.pdf

It prints per-page timings, the peak RSS, and the first lines of each page, so
you can see reading order and table structure with your own eyes rather than
inferring them from a green test suite — which is what docs/agent-brief.md asks
for.
"""
import resource
import sys
import time
from pathlib import Path

import extract as extractor


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"not a file: {path}")
        return 2

    print(f"file      {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    started = time.monotonic()
    try:
        result = extractor.extract(str(path), path.name)
    except Exception as err:
        # Loud here, unlike the worker: this is the operator's own machine and
        # the whole point is to see what actually broke.
        print(f"FAILED    {type(err).__name__}: {err}")
        raise
    elapsed = time.monotonic() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    pages = result["pages"]
    chars = sum(len(p["text"]) for p in pages)
    print(f"pages     {len(pages)} of {result['total']}")
    print(f"time      {elapsed:.1f}s total, {elapsed / max(1, len(pages)):.2f}s/page")
    print(f"peak RSS  {peak:.0f} MB   (models load on the first convert, so a")
    print(f"          second run of the same file is the steady-state number)")
    print(f"text      {chars} characters")
    print(f"status    " + ", ".join(f"{s}={sum(1 for p in pages if p['status'] == s)}"
                                    for s in sorted({p["status"] for p in pages})))
    if any("|" in p["text"] for p in pages):
        print("tables    Markdown pipe tables present — table structure survived")
    else:
        print("tables    none detected (fine if the document has none)")
    print()
    for page in pages[:5]:
        head = " / ".join(page["text"].splitlines()[:3])[:160]
        print(f"  [{page['number']:>3}] {page['status']:<9} {head}")
    if len(pages) > 5:
        print(f"  … {len(pages) - 5} more")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
