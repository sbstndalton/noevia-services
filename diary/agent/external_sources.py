"""External diary-like source folders: detection and manual import support.

Operators point DIARY_EXTERNAL_SOURCES (comma-separated absolute paths) at
folders of pre-existing plain-text journal files. This module READS those
folders only — it never writes, moves, renames, or deletes anything in them.
Import is a separate, explicit, one-file-at-a-time action (the endpoint lives
in app.py); it copies the file's text into the corpus via the normal journal
pipeline, leaving the original untouched.

File types: .txt, .md, .markdown (deliberately narrow — see the phase notes).
Dates: filename patterns first (YYYY-MM-DD, then YYYYMMDD), file mtime as the
fallback. Every reported date says which method produced it so nothing is
hidden from the user.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown"}

# Safety valves: a misconfigured path pointing at a huge tree must not hang
# the sidecar or produce an unbounded response.
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
MAX_FILES_PER_SOURCE = 500

_FILENAME_DATE_PATTERNS = (
    # 2026-09-06 anywhere in the name (leading/separated by non-digits).
    re.compile(r"(?:^|\D)(\d{4})-(\d{1,2})-(\d{1,2})(?:\D|$)"),
    # 20260906 anywhere in the name.
    re.compile(r"(?:^|\D)(\d{4})(\d{2})(\d{2})(?:\D|$)"),
)


def external_source_paths(env: Optional[dict] = None) -> List[str]:
    """Configured absolute source paths, deduplicated, in declared order.

    Read per request (not at import time) so tests can monkeypatch the env and
    operators can change the list without code changes.
    """
    raw = (env or os.environ).get("DIARY_EXTERNAL_SOURCES", "")
    paths: List[str] = []
    seen = set()
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        resolved = str(Path(p).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)
    return paths


def infer_date(path: Path, env: Optional[dict] = None) -> tuple:
    """Best-guess date for a source file.

    Returns (date_or_None, source) where source is "filename" or "mtime".
    An unparseable filename date falls through to the next pattern, then mtime.
    """
    name = path.name
    for rx in _FILENAME_DATE_PATTERNS:
        m = rx.search(name)
        if not m:
            continue
        try:
            year, month, day = (int(g) for g in m.groups())
            return datetime(year, month, day).date(), "filename"
        except ValueError:
            continue  # matched shape but not a real calendar date
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime).date(), "mtime"
    except OSError:
        return None, "unavailable"


def scan_source(source_path: str) -> dict:
    """List diary-like files under one configured source path (read-only).

    Recursive, capped at MAX_FILES_PER_SOURCE. Missing paths are reported,
    not raised: a configured folder that isn't mounted yet is a normal state,
    not an error the caller should crash on.
    """
    root = Path(source_path)
    if not root.exists():
        return {"path": source_path, "exists": False, "files": [], "total": 0, "truncated": False}
    if not root.is_dir():
        return {
            "path": source_path,
            "exists": True,
            "error": "not a directory",
            "files": [],
            "total": 0,
            "truncated": False,
        }

    files: List[dict] = []
    truncated = False
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            if len(files) >= MAX_FILES_PER_SOURCE:
                truncated = True
                break
            full = Path(dirpath) / fname
            try:
                stat = full.stat()
            except OSError:
                continue  # vanished mid-scan; skip rather than fail the listing
            date, date_source = infer_date(full)
            files.append({
                "name": fname,
                "rel_path": str(full.relative_to(root)),
                "size": stat.st_size,
                "date": date.isoformat() if date else None,
                "date_source": date_source,
            })
        if truncated:
            break

    files.sort(key=lambda f: (f["date"] is None, f["date"] or "", f["rel_path"]))
    return {
        "path": source_path,
        "exists": True,
        "files": files,
        "total": len(files),
        "truncated": truncated,
    }


def resolve_import_target(source_path: str, rel_path: str) -> Optional[Path]:
    """Safely resolve a listed relative path under a configured source.

    Returns the resolved Path, or None when anything smells wrong: absolute
    or parent-traversing rel_paths, paths that escape the source root, or
    files that no longer exist. The source itself must be one of the exact
    configured paths — callers check that against external_source_paths().
    """
    if not source_path or not rel_path:
        return None
    root = Path(source_path)
    candidate = Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    full = (root / candidate).resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError:
        return None
    if not full.is_file():
        return None
    if full.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None
    return full
