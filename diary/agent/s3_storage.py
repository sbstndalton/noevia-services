"""S3-compatible corpus backend with guarded conditional writes.

Design (irreplaceable-data discipline, mirroring webdav.py):
  - Writes are conditional: the backend re-GETs the current object + ETag before
    every PUT attempt, then sends the PUT with If-Match (overwrite) or
    If-None-Match:"*" (create-only). On 412/409 (or 404 on a guarded overwrite,
    i.e. the object vanished), it re-reads and retries, bounded — the
    read-verify-write loop required for stores whose conditional-write support
    varies (some S3-compatible servers accept the headers but ignore them).
  - Bytes never written blind: every PUT carries a precondition, and every
    precondition is verified against a fresh read, never a cached ETag alone.
  - Journal replay safety comes from CorpusStore (xid dedupe on append), so
    replays after a crash mid-write are idempotent exactly as for local/WebDAV.

Addressing: path-style only (endpoint/bucket/key) — the form every
self-hosted S3-compatible server (MinIO, Garage, SeaweedFS, B2's S3 API)
speaks, and still accepted by AWS S3. Virtual-host style is deliberately not
implemented; point endpoint_url at the region host and use path-style.
Credentials are optional: without them requests are sent unsigned (works for
read-only public buckets and servers that auth by network position).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import logging
import time
from typing import Dict, Optional, Tuple
from urllib.parse import quote, urlparse
from xml.etree import ElementTree

import httpx

from .util import ensure_not_redirect, make_client

log = logging.getLogger(__name__)

ALGORITHM = "AWS4-HMAC-SHA256"


def clean_etag(etag: Optional[str]) -> Optional[str]:
    """Normalize an ETag to the quoted form used in If-Match comparisons."""
    if not etag:
        return etag
    core = etag.strip().strip('"')
    return f'"{core}"'


def _uri_encode(value: str) -> str:
    """AWS canonical URI/query encoding: RFC3986, '/' kept in paths only."""
    return quote(value, safe="-_.~")


class S3CorpusBackend:
    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        access_key: str = "",
        secret_key: str = "",
        region: str = "us-east-1",
        prefix: str = "",
        timeout_s: float = 60.0,
        session_token: str = "",
    ):
        parsed = urlparse(endpoint_url if "://" in endpoint_url else f"https://{endpoint_url}")
        self.scheme = parsed.scheme or "https"
        self.host = parsed.netloc
        self.base = f"{self.scheme}://{self.host}"
        self.bucket = bucket.strip("/")
        self.prefix = prefix.strip("/")
        self.region = region or "us-east-1"
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.timeout_s = timeout_s
        self._client = make_client(base_url=self.base, timeout_s=timeout_s)
        self._etag_cache: Dict[str, Optional[str]] = {}

    # ---------------- paths ----------------

    def _key(self, remote_path: str) -> str:
        clean = remote_path.lstrip("/")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def _path(self, key: str) -> str:
        return "/" + "/".join(_uri_encode(seg) for seg in f"{self.bucket}/{key}".split("/"))

    # ---------------- SigV4 ----------------

    @staticmethod
    def _sha256_hex(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _signed_headers(
        self,
        method: str,
        path: str,
        query: Optional[dict],
        payload: bytes,
        extra: Optional[dict] = None,
        amzdate: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build signed request headers. `amzdate` overrides the timestamp
        (test vectors); `extra` headers are signed and sent verbatim."""
        now = _dt.datetime.now(_dt.timezone.utc)
        amzdate = amzdate or now.strftime("%Y%m%dT%H%M%SZ")
        datestamp = amzdate[:8]
        payload_hash = self._sha256_hex(payload)

        headers = {
            "host": self.host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amzdate,
        }
        if self.session_token:
            headers["x-amz-security-token"] = self.session_token
        for k, v in (extra or {}).items():
            headers[k.lower()] = str(v)

        canonical_headers = "".join(f"{k}:{headers[k].strip()}\n" for k in sorted(headers))
        signed_headers = ";".join(sorted(headers))

        canonical_query = "&".join(
            f"{_uri_encode(k)}={_uri_encode(str(v))}" for k, v in sorted((query or {}).items())
        )
        canonical_request = "\n".join([method, path, canonical_query, canonical_headers, signed_headers, payload_hash])

        scope = f"{datestamp}/{self.region}/s3/aws4_request"
        string_to_sign = "\n".join([
            ALGORITHM, amzdate, scope, self._sha256_hex(canonical_request.encode("utf-8")),
        ])

        def _hmac(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        signing_key = _hmac(_hmac(_hmac(_hmac(f"AWS4{self.secret_key}".encode(), datestamp), self.region), "s3"), "aws4_request")
        signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        merged = {k.title(): v for k, v in (extra or {}).items()}
        merged.update({
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amzdate,
            "Authorization": (
                f"{ALGORITHM} Credential={self.access_key}/{scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            ),
        })
        if self.session_token:
            merged["x-amz-security-token"] = self.session_token
        return merged

    def _request(
        self,
        method: str,
        key: str,
        *,
        query: Optional[dict] = None,
        headers: Optional[dict] = None,
        data: bytes = b"",
    ) -> httpx.Response:
        path = self._path(key)
        if not self.secret_key:
            # Unsigned mode: no credentials configured (public/position-authed buckets).
            resp = self._client.request(method, path, params=query, content=data, headers=dict(headers or {}))
        else:
            signed = self._signed_headers(method, path, query, data, headers)
            resp = self._client.request(method, path, params=query, content=data, headers=signed)
        # User-configured endpoint: refuse bounced requests (see util.ensure_not_redirect).
        ensure_not_redirect(resp)
        return resp

    # ---------------- primitives ----------------

    def get(self, remote_path: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Fetch object bytes + ETag. Returns (None, None) if the object is absent."""
        resp = self._request("GET", self._key(remote_path))
        if resp.status_code == 404:
            self._etag_cache[remote_path] = None
            return None, None
        resp.raise_for_status()
        etag = clean_etag(resp.headers.get("ETag"))
        self._etag_cache[remote_path] = etag
        return resp.content, etag

    def get_text(self, remote_path: str) -> Tuple[Optional[str], Optional[str]]:
        data, etag = self.get(remote_path)
        if data is None:
            return None, None
        return data.decode("utf-8", errors="replace"), etag

    def put(
        self,
        remote_path: str,
        data: bytes,
        if_match: Optional[str] = None,
        if_none_match: str = "*",
        max_retries: int = 5,
    ) -> Tuple[bool, Optional[str], int]:
        """Conditional PUT.

        - if_match given   -> overwrite only if the stored object still matches (412 => conflict).
        - if_none_match='* -> create-only (412 => someone created it concurrently).
        Each attempt re-reads the stored object and verifies the precondition holds
        before sending — read-verify-write, so a server that ignores conditional
        headers still cannot lose an update silently.
        Returns (ok, new_etag, status_code).
        """
        headers_base = {"Content-Type": "text/markdown; charset=utf-8"}
        status = 0
        for attempt in range(max_retries):
            current, current_etag = self.get(remote_path)
            if if_match is not None and current_etag != if_match:
                status = 412  # stale precondition — re-read, re-verify next loop
            elif if_match is None and if_none_match == "*" and current is not None:
                status = 412  # lost a create race
            else:
                headers = dict(headers_base)
                if if_match is not None:
                    headers["If-Match"] = if_match
                else:
                    headers["If-None-Match"] = if_none_match
                resp = self._request("PUT", self._key(remote_path), data=data, headers=headers)
                if resp.status_code in (200, 201, 204):
                    etag = clean_etag(resp.headers.get("ETag")) or f'"{self._sha256_hex(data)}"'
                    self._etag_cache[remote_path] = etag
                    return True, etag, resp.status_code
                status = resp.status_code
                if status not in (412, 409, 405):
                    resp.raise_for_status()  # unexpected error — surface it
            time.sleep(0.2 * (attempt + 1))
        return False, None, status

    def exists(self, remote_path: str) -> bool:
        resp = self._request("HEAD", self._key(remote_path))
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return False

    def list_dir(self, remote_dir: str) -> list:
        """List-V2 with delimiter — returns [{'name', 'path', 'etag', 'lastmod', 'is_dir'}].

        Paths are returned relative to the configured prefix, matching what
        get/put expect for the same entry.
        """
        directory = remote_dir.strip("/")
        full = f"{self.prefix}/{directory}" if self.prefix else directory
        query = {"list-type": "2", "prefix": f"{full}/" if full else "", "delimiter": "/", "max-keys": "1000"}
        resp = self._request("GET", "", query=query)
        if resp.status_code == 404:  # bucket-level listing never 404s; defensive
            return []
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        ns = {"s": root.tag.split("}")[0].lstrip("{")} if root.tag.startswith("{") else {}

        def _find(node, name):
            return node.find(f"s:{name}", ns) if ns else node.find(name)

        def _all(node, name):
            return node.findall(f"s:{name}", ns) if ns else node.findall(name)

        prefix_stripped = f"{self.prefix}/" if self.prefix else ""
        entries = []
        # ListBucketResult holds one <Contents>/<CommonPrefixes> wrapper PER entry,
        # so findall (not find) — find would silently return only the first.
        for cp in _all(root, "CommonPrefixes"):
            node = _find(cp, "Prefix")
            full_dir = (node.text or "").rstrip("/") if node is not None else ""
            rel = full_dir[len(prefix_stripped):] if prefix_stripped and full_dir.startswith(prefix_stripped) else full_dir
            if not rel:
                continue
            entries.append({
                "name": rel.rsplit("/", 1)[-1],
                "path": rel,
                "etag": None,
                "lastmod": None,
                "is_dir": True,
            })
        for c in _all(root, "Contents"):
            key_node = _find(c, "Key")
            full_key = (key_node.text or "") if key_node is not None else ""
            if not full_key or full_key.endswith("/"):
                continue
            rel = full_key[len(prefix_stripped):] if prefix_stripped and full_key.startswith(prefix_stripped) else full_key
            name = rel.rsplit("/", 1)[-1]
            parent = rel[: -len(name) - 1] if "/" in rel else ""
            if parent != directory:  # keep only direct children of the requested dir
                continue
            etag_node = _find(c, "ETag")
            lm_node = _find(c, "LastModified")
            entries.append({
                "name": name,
                "path": rel,
                "etag": clean_etag(etag_node.text if etag_node is not None else None),
                "lastmod": lm_node.text if lm_node is not None else None,
                "is_dir": False,
            })
        entries.sort(key=lambda item: item["name"].lower())
        return entries

    def close(self) -> None:
        self._client.close()
