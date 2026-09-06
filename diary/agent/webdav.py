"""Corpus access layer: WebDAV client with ETag-guarded atomic writes.

Design (irreplaceable-data discipline):
  - Writes are conditional: GET current content + ETag, build new content, PUT with If-Match.
  - On 412 Precondition Failed (concurrent change), re-GET and retry, bounded.
  - Last-modified maps are cached per (path, etag) to avoid GETs on every append.
  - Bytes never written blind: every PUT carries a precondition.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import httpx

from .util import make_client

log = logging.getLogger(__name__)


def clean_etag(etag: Optional[str]) -> Optional[str]:
    """Normalize an ETag for If-Match comparisons.

    Apache mod_deflate appends a compression-variant suffix ("...-gzip" / "...-br")
    to ETags on compressed responses. Nextcloud compares If-Match against the plain
    ETag, so such a value never matches — every conditional PUT would 412 forever.
    The client now also requests uncompressed responses (see util.make_client);
    this normalization is defense in depth for any cached/older value.
    """
    if not etag:
        return etag
    core = etag.strip().strip('"')
    if core.endswith("-gzip") or core.endswith("-br"):
        core = core.rsplit("-", 1)[0]
    return f'"{core}"'


class WebDAVCorpusBackend:
    def __init__(self, base_url: str, username: str, password: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/") + "/"
        self.base_path = unquote(urlparse(self.base_url).path).rstrip("/") + "/"
        self.auth = (username, password) if username else None
        self.timeout_s = timeout_s
        self._client = make_client(base_url=self.base_url, timeout_s=timeout_s, auth=self.auth)
        self._etag_cache: Dict[str, Optional[str]] = {}
        self._lm_cache: Dict[str, Optional[str]] = {}

    # ---------------- path helpers ----------------

    def _path(self, remote_path: str) -> str:
        """URL-encode each path segment, preserving / separators."""
        clean = remote_path.lstrip("/")
        return "/".join(quote(seg) for seg in clean.split("/"))

    def _url(self, remote_path: str) -> str:
        return self.base_url + self._path(remote_path)

    # ---------------- primitives ----------------

    def get(self, remote_path: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Fetch file bytes + ETag. Returns (None, None) if the file does not exist."""
        url = self._url(remote_path)
        resp = self._client.get(url, headers={"Accept": "*/*"})
        if resp.status_code == 404:
            return None, None
        resp.raise_for_status()
        etag = clean_etag(resp.headers.get("ETag"))
        self._etag_cache[remote_path] = etag
        self._lm_cache[remote_path] = resp.headers.get("Last-Modified")
        return resp.content, etag

    def get_text(self, remote_path: str) -> Tuple[Optional[str], Optional[str]]:
        data, etag = self.get(remote_path)
        if data is None:
            return None, None
        return data.decode("utf-8", errors="replace"), etag

    def _ensure_parent_dirs(self, remote_path: str) -> None:
        """Create missing WebDAV collections from shallowest to deepest.

        MKCOL is idempotent here: Nextcloud returns 405 when a collection
        already exists, which is treated as success. This is needed by the
        daily layout when the first entry of a new month or year is written.
        """
        parts = remote_path.strip("/").split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directory = "/".join(parts[:index])
            resp = self._client.request("MKCOL", self._url(directory + "/"))
            if resp.status_code in (201, 405):
                continue
            resp.raise_for_status()

    def put(
        self,
        remote_path: str,
        data: bytes,
        if_match: Optional[str] = None,
        if_none_match: str = "*",
        max_retries: int = 5,
    ) -> Tuple[bool, Optional[str], int]:
        """Conditional PUT.

        - if_match given  -> overwrite only if remote still matches that ETag (412 => conflict).
        - if_none_match='*' -> create-only (405/412 => someone created it concurrently).
        Returns (ok, new_etag, status_code).
        """
        url = self._url(remote_path)
        headers: Dict[str, str] = {"Content-Type": "text/markdown; charset=utf-8"}
        if if_match is not None:
            headers["If-Match"] = if_match
        else:
            headers["If-None-Match"] = if_none_match
            self._ensure_parent_dirs(remote_path)

        for attempt in range(max_retries):
            resp = self._client.put(url, content=data, headers=headers)
            if resp.status_code in (200, 201, 204):
                etag = clean_etag(resp.headers.get("ETag"))
                self._etag_cache[remote_path] = etag
                return True, etag, resp.status_code
            if resp.status_code in (412, 409) or (if_match is None and resp.status_code == 405):
                # Precondition failed or conflict: caller may re-GET and retry; we back off lightly.
                import time

                time.sleep(0.2 * (attempt + 1))
                continue
            resp.raise_for_status()  # unexpected error — surface it

        return False, None, resp.status_code

    def exists(self, remote_path: str) -> bool:
        url = self._url(remote_path)
        resp = self._client.head(url)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return False

    def list_dir(self, remote_dir: str) -> list:
        """PROPFIND (Depth 1) — returns [{'name', 'path', 'etag', 'lastmod', 'is_dir'}]."""
        url = self._url(remote_dir.rstrip("/") + "/")
        resp = self._client.request(
            "PROPFIND",
            url,
            headers={"Depth": "1", "Content-Type": "application/xml"},
            content='<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:getetag/><d:getlastmodified/><d:resourcetype/></d:prop></d:propfind>',
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        body = resp.text
        entries: list = []
        # Lightweight XML extraction for the flat Depth-1 response shape.
        for m in re.finditer(r"<d:response>(.*?)</d:response>", body, re.S | re.I):
            block = m.group(1)
            href_m = re.search(r"<d:href>(.*?)</d:href>", block, re.S | re.I)
            if not href_m:
                continue
            href = href_m.group(1)
            href_path = unquote(urlparse(href).path)
            if not href_path.startswith(self.base_path):
                continue
            path = href_path[len(self.base_path):]
            name = path.rstrip("/").rsplit("/", 1)[-1]
            if not name:
                continue
            # keep only direct children of the requested dir
            parent = path.rstrip("/")
            parent_dir = parent.rsplit("/", 1)[0] if "/" in parent else ""
            if parent_dir != remote_dir.strip("/"):
                continue
            # Case-insensitive, consistent with the rest of the XML parsing.
            is_dir = re.search(r"<d:collection\s*/?>", block, re.I) is not None
            etag_m = re.search(r"<d:getetag[^>]*>(&quot;)?([^<&]*)", block, re.I)
            etag = etag_m.group(2) if etag_m else None
            lm_m = re.search(r"<d:getlastmodified>(.*?)</d:getlastmodified>", block, re.S | re.I)
            entries.append(
                {
                    "name": name,
                    "path": path.rstrip("/") if is_dir else path,
                    "etag": etag,
                    "lastmod": lm_m.group(1) if lm_m else None,
                    "is_dir": is_dir,
                }
            )
        return entries

    def close(self) -> None:
        self._client.close()


# Compatibility import for one release.
WebDAVClient = WebDAVCorpusBackend
