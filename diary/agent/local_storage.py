"""Safe local-folder implementation of the diary corpus storage contract."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from .dedicated_storage import check_volume


class LocalCorpusBackend:
    def __init__(self, root: str, volume_identity: str | None = None, reader_uid: int | None = None):
        self.root = Path(root).expanduser().resolve()
        self.volume_identity = volume_identity
        self.reader_uid = reader_uid
        if volume_identity:
            check_volume(self.root, volume_identity)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[Path, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _path(self, relative: str) -> Path:
        if self.volume_identity:
            check_volume(self.root, self.volume_identity)
        candidate = (self.root / relative.lstrip("/")).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"corpus path escapes configured root: {relative!r}")
        return candidate

    def _lock(self, path: Path) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(path, threading.Lock())

    @staticmethod
    def _version(data: bytes) -> str:
        return f'"sha256:{hashlib.sha256(data).hexdigest()}"'

    def get(self, path: str) -> Tuple[Optional[bytes], Optional[str]]:
        target = self._path(path)
        try:
            data = target.read_bytes()
        except FileNotFoundError:
            return None, None
        return data, self._version(data)

    def get_text(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        data, version = self.get(path)
        return (None, None) if data is None else (data.decode("utf-8", errors="replace"), version)

    def put(
        self,
        path: str,
        data: bytes,
        if_match: Optional[str] = None,
        if_none_match: str = "*",
        max_retries: int = 5,
    ) -> Tuple[bool, Optional[str], int]:
        del max_retries  # retries happen in CorpusStore after it re-reads fresh content
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(target):
            current, current_version = self.get(path)
            if if_match is not None and current_version != if_match:
                return False, None, 412
            if if_match is None and if_none_match == "*" and current is not None:
                return False, None, 412

            fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    # Atomic replacement must retain access for the dedicated
                    # SMB reader even when the companion runs as root.
                    if self.reader_uid is not None:
                        os.fchown(handle.fileno(), self.reader_uid, -1)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, target)
                dir_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            finally:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
        return True, self._version(data), 201 if current is None else 204

    def create_directory(self, path: str) -> None:
        """Create exactly one collection; never create missing ancestors or replace."""
        target = self._path(path)
        with self._lock(target):
            target.mkdir(mode=0o755, parents=False, exist_ok=False)
            parent_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)

    def exists(self, path: str) -> bool:
        return self._path(path).is_file()

    def list_dir(self, path: str) -> list:
        directory = self._path(path)
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise NotADirectoryError(path)
        entries = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            # Do not follow symlinks outside the configured root.
            resolved = child.resolve()
            if resolved != self.root and self.root not in resolved.parents:
                continue
            is_dir = child.is_dir()
            data = None if is_dir else child.read_bytes()
            entries.append({
                "name": child.name,
                "path": "/".join(part for part in [path.strip("/"), child.name] if part),
                "etag": None if data is None else self._version(data),
                "lastmod": child.stat().st_mtime,
                "is_dir": is_dir,
            })
        return entries

    def close(self) -> None:
        return None
