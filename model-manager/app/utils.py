from __future__ import annotations

import re


def human_bytes(n: float) -> str:
    step = 1024.0
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < step:
            return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} EB"


_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})(?=\.[^.]+$)")


def shard_key(filename: str) -> tuple[str, int | None, int | None]:
    """Return (base_name_without_shard_suffix, part_index, part_total) or (name, None, None)."""
    m = _SHARD_RE.search(filename)
    if not m:
        return filename, None, None
    return _SHARD_RE.sub("", filename), int(m.group(1)), int(m.group(2))


# noevia: model folders may be flat, one directory per model, or Hugging Face's download
# cache (models--owner--repo/snapshots/<revision>/file.gguf, where the file is a link into
# blobs/). Walk a few levels and skip the cache internals.
_SKIP_DIRS = {"blobs", "refs", ".locks", ".cache", ".git", "__pycache__"}
MAX_MODEL_DEPTH = 4


def iter_gguf(root):
    """Yield (relative_dir, Path) for every .gguf under root, up to MAX_MODEL_DEPTH folders deep."""
    import os
    from pathlib import Path
    root = Path(root)
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        depth = len(here.parts) - base_depth
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS) if depth < MAX_MODEL_DEPTH else []
        rel = here.relative_to(root).as_posix()
        for name in sorted(filenames):
            if name.lower().endswith(".gguf"):
                f = here / name
                if f.is_file():
                    yield ("" if rel == "." else rel), f
