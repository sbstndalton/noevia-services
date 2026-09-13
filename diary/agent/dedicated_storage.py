"""Operator-owned tenant volumes, independent of general storage connections."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path, PurePosixPath


class StorageUnavailable(OSError):
    """A configured corpus must never fall back to an empty local directory."""


def check_volume(root: Path, identity: str) -> None:
    try:
        marker = root / ".noevia-diary-volume"
        if not root.is_dir() or marker.is_symlink() or marker.read_text().strip() != identity:
            raise ValueError("volume identity mismatch")
    except (OSError, ValueError) as exc:
        raise StorageUnavailable("Dedicated Diary storage is unavailable; restore its configured volume.") from exc


def tenant_volume(user_id: str) -> dict | None:
    """Only operator environment can choose host paths, never a client header."""
    try:
        volumes = json.loads(os.environ.get("DIARY_LOCAL_VOLUMES") or "{}")
        if not isinstance(volumes, dict):
            raise ValueError("expected object")
        roots = set()
        for tenant, volume in volumes.items():
            if str(uuid.UUID(tenant)) != tenant or not isinstance(volume, dict):
                raise ValueError("invalid tenant")
            root = Path(volume["root"])
            prefix = volume["prefix"]
            reader_uid = volume.get("reader_uid")
            if reader_uid is not None and (type(reader_uid) is not int or reader_uid < 0):
                raise ValueError("invalid reader uid")
            if not root.is_absolute() or not isinstance(prefix, str):
                raise ValueError("invalid root or prefix")
            if prefix and (PurePosixPath(prefix).is_absolute() or any(p in ("", ".", "..") for p in prefix.split("/")) or "\\" in prefix):
                raise ValueError("invalid prefix")
            resolved = root.resolve()
            if any(resolved == other or resolved in other.parents or other in resolved.parents for other in roots):
                raise ValueError("tenant volumes overlap")
            roots.add(resolved)
        volume = volumes.get(user_id)
        if volume is not None:
            check_volume(Path(volume["root"]), user_id)
        return volume
    except StorageUnavailable:
        raise
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise StorageUnavailable("Dedicated Diary storage configuration is invalid.") from exc
