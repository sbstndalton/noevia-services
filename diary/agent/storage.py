"""Storage contract and backend construction for diary corpus files."""
from __future__ import annotations

from typing import Optional, Protocol, Tuple

from .config import Config


class CorpusBackend(Protocol):
    """Minimal versioned file-store contract used by :class:`CorpusStore`."""

    def get_text(self, path: str) -> Tuple[Optional[str], Optional[str]]: ...

    def put(
        self,
        path: str,
        data: bytes,
        if_match: Optional[str] = None,
        if_none_match: str = "*",
        max_retries: int = 5,
    ) -> Tuple[bool, Optional[str], int]: ...

    def exists(self, path: str) -> bool: ...

    def list_dir(self, path: str) -> list: ...

    def close(self) -> None: ...


def create_backend(cfg: Config) -> CorpusBackend:
    kind = str(cfg.get("corpus.backend", "local")).strip().lower()
    if kind == "local":
        from .local_storage import LocalCorpusBackend

        return LocalCorpusBackend(cfg.get("corpus.local.root") or "data/corpus")
    if kind == "webdav":
        from .webdav import WebDAVCorpusBackend

        return WebDAVCorpusBackend(
            base_url=cfg.get("corpus.webdav.base_url"),
            username=cfg.get("corpus.webdav.username") or "",
            password=cfg.get("corpus.webdav.password") or "",
            timeout_s=float(cfg.get("corpus.webdav.timeout_s", 60)),
        )
    raise ValueError(f"unsupported corpus backend: {kind!r} (expected 'local' or 'webdav')")
