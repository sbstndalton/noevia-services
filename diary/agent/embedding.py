"""Embedding profile, configuration and client for the Diary index (#1636 / #1638).

Mirrors noevia-core's embedding-profile.cjs and embedding-config.cjs:

* A **profile** frames text before it is embedded. ``nomic-v1`` (the default) sends text
  unchanged, exactly as the Diary always has (no task prefixes), so every existing index was built
  that way. ``embeddinggemma-2`` uses the model card's prompts: queries
  ``task: search result | query: {text}``, stored chunks ``title: none | text: {text}``.
* The **configuration** holds ``active`` (search and indexing), ``next`` (what a re-index is
  building) and ``previous`` (kept for rollback). A target is (base_url, model, profile); an empty
  base_url means the chat engine (``llm.base_url``, with its key); a base_url means a dedicated
  embeddings server, which never receives the engine key. The identity of stored vectors is
  (profile, model). The environment (LLM_EMBED_BASE_URL, LLM_EMBED_MODEL, LLM_EMBED_PROFILE and
  LLM_EMBED_NEXT_*) only seeds it; a runtime change (embedding_migration, the one path that changes
  the model or the endpoint) is saved to ``embedding-config.json`` beside the index and wins on the
  next start. Deleting that file returns to the environment.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

DEFAULT_PROFILE = "nomic-v1"
PROFILES = ("nomic-v1", "embeddinggemma-2")
KINDS = ("query", "document", "similarity")
_MODEL_RE = re.compile(r"^[^\s\x00-\x1f]{1,200}$")


# A start-up check that cannot get an answer (a resolver that is not up yet, a compose service that
# is still starting) is retried for about half a minute, then re-checked every minute; only a public
# answer or a malformed address is a permanent refusal.
RETRY_DELAYS_S = (1.0, 2.0, 4.0, 8.0, 15.0)
RECHECK_S = 60.0


class EmbeddingConfigError(ValueError):
    def __init__(self, message: str, code: str = "invalidTarget", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def clean_title(title: Optional[str]) -> str:
    t = re.sub(r"\s+", " ", re.sub(r"[\r\n|]+", " ", str(title or ""))).strip()[:160]
    return t or "none"


def frame(profile: str, texts: Sequence[str], kind: Union[str, Sequence[str]] = "document",
          titles: Optional[Sequence[Optional[str]]] = None) -> List[str]:
    """The texts to send for ``profile``. Raises on an unknown profile or kind (fail closed)."""
    if profile not in PROFILES:
        raise EmbeddingConfigError(f'Unknown embedding profile "{profile}" (known: {", ".join(PROFILES)})', "unknownProfile")
    out = []
    for i, text in enumerate(texts):
        k = kind[i] if isinstance(kind, (list, tuple)) else kind
        if k not in KINDS:
            raise EmbeddingConfigError(f'Unknown embedding kind "{k}"', "unknownKind")
        text = str(text if text is not None else "")
        if profile == "nomic-v1":
            out.append(text)
        elif k == "query":
            out.append(f"task: search result | query: {text}")
        elif k == "similarity":
            out.append(f"task: sentence similarity | query: {text}")
        else:
            out.append(f"title: {clean_title(titles[i] if titles else None)} | text: {text}")
    return out


# ---- #1759: a document longer than the embeddings server takes in one batch ----
#
# The server refuses an input with more tokens than its batch (llama.cpp ``-ub``) or per-slot context.
# The Diary's chunk is a whole ``###`` subsection, which can be longer than that, and EmbeddingGemma 2
# counts about 9% more tokens than nomic for the same text. A document the server refuses as too long
# is cut into pieces at paragraph, line, sentence or word boundaries (sized by the token counts the
# server reported, with a safety margin, or halved when it reported none), each piece is framed and
# embedded on its own, and the chunk's vector is the length-weighted mean of the piece vectors,
# normalised. The chunk stays the unit of retrieval and storage (one row, one vector), so search
# results, the index schema and the per-model slots are unchanged. A text the server accepts is sent
# exactly as before: nothing is split ahead of time, under any model or profile. Queries are never split.
SPLIT_MARGIN = 0.85       # pieces aim at this share of the limit the server reported
SPLIT_MAX_DEPTH = 6       # a piece refused again is cut again, at most this many times
SPLIT_MIN_CHARS = 64      # a piece this short that is still refused is a real refusal


def split_text(text: str, parts: int) -> List[str]:
    """``text`` cut into about ``parts`` pieces of similar length, at the best boundary near each cut."""
    parts = max(2, int(parts))
    size = len(text) / parts
    pieces: List[str] = []
    start = 0
    for i in range(1, parts):
        ideal = int(i * size)
        if ideal <= start:
            continue
        lo = max(start + 1, ideal - int(size * 0.4))
        cut = ideal
        for sep in ("\n\n", "\n", ". ", " "):
            at = text.rfind(sep, lo, ideal + 1)
            if at > start:
                cut = at + len(sep)
                break
        piece = text[start:cut]
        if piece.strip():
            pieces.append(piece)
        start = cut
    rest = text[start:]
    if rest.strip():
        pieces.append(rest)
    return pieces if len(pieces) > 1 else [text[: len(text) // 2], text[len(text) // 2:]]


def _mean(vectors: List[List[float]], weights: List[int]) -> List[float]:
    total = float(sum(weights)) or 1.0
    out = [sum(v[j] * w for v, w in zip(vectors, weights)) / total for j in range(len(vectors[0]))]
    norm = sum(x * x for x in out) ** 0.5
    return [x / norm for x in out] if norm > 0 else out


def embed_fitting(send: Callable[[List[str], List[str], List[Optional[str]]], List[List[float]]],
                  texts: Sequence[str], kind: Union[str, Sequence[str]] = "document",
                  titles: Optional[Sequence[Optional[str]]] = None, _depth: int = 0) -> List[List[float]]:
    """``send(texts, kinds, titles)`` for ``texts``, re-splitting any document the server refuses as
    too long (see above). Any other error, and a refused query, is raised unchanged."""
    from .llm import InputTooLarge  # local import: llm imports nothing from here

    texts = [str(t if t is not None else "") for t in texts]
    kinds = [kind[i] if isinstance(kind, (list, tuple)) else kind for i in range(len(texts))]
    tlist = [titles[i] if titles else None for i in range(len(texts))]
    try:
        return send(texts, kinds, tlist)
    except InputTooLarge as exc:
        if len(texts) > 1:  # find the one that is too long; the others go as they are
            out: List[List[float]] = []
            for t, k, ti in zip(texts, kinds, tlist):
                out.extend(embed_fitting(send, [t], [k], [ti], _depth))
            return out
        text, k = texts[0], kinds[0]
        if k != "document" or _depth >= SPLIT_MAX_DEPTH or len(text) < SPLIT_MIN_CHARS:
            raise
        parts = 2
        if exc.tokens and exc.limit and exc.limit > 0:
            parts = max(2, -(-exc.tokens // max(1, int(exc.limit * SPLIT_MARGIN))))
        pieces = split_text(text, parts)
        log.info("embedding: a document of %d characters was too long for the server (%s); embedding it in %d pieces",
                 len(text), f"{exc.tokens} tokens, limit {exc.limit}" if exc.tokens else "no token count", len(pieces))
        vectors = [embed_fitting(send, [p], [k], [tlist[0]], _depth + 1)[0] for p in pieces]
        return [_mean(vectors, [len(p) for p in pieces])]


@dataclass(frozen=True)
class Target:
    base_url: Optional[str]
    model: str
    profile: str = DEFAULT_PROFILE

    @property
    def identity(self) -> str:
        return f"{self.profile}\0{self.model}"

    @property
    def slot_id(self) -> str:
        return hashlib.sha256(self.identity.encode("utf-8")).hexdigest()[:16]

    def to_json(self) -> Dict[str, Any]:
        return {"base_url": self.base_url, "model": self.model, "profile": self.profile}


def _normalize_base_url(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    m = re.match(r"^(https?)://([^/?#@\s]+)(/[^?#\s]*)?$", text)
    if not m:
        raise EmbeddingConfigError("The embeddings address must be a plain http(s) URL, with no credentials, query or fragment.")
    return text.rstrip("/")


def normalize_target(raw: Any) -> Target:
    if isinstance(raw, Target):
        raw = raw.to_json()
    if not isinstance(raw, dict):
        raise EmbeddingConfigError("An embedding target needs a model.")
    model = str(raw.get("model") or "").strip()
    if not _MODEL_RE.match(model):
        raise EmbeddingConfigError("An embedding target needs a model name (no spaces, at most 200 characters).")
    profile = str(raw.get("profile") or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE
    if profile not in PROFILES:
        raise EmbeddingConfigError(f'Unknown embedding profile "{profile}" (known: {", ".join(PROFILES)}).')
    base = raw.get("base_url", raw.get("baseUrl"))
    return Target(_normalize_base_url(base), model, profile)


def seed_from_cfg(cfg) -> Dict[str, Optional[Target]]:
    """Targets the configuration (environment) describes. An unknown profile is kept, so it fails closed later."""
    def get(key: str) -> str:
        return str(cfg.get(key) or "").strip()
    active = Target(get("llm.embed_base_url").rstrip("/") or None, get("llm.embed_model") or "default", get("llm.embed_profile") or DEFAULT_PROFILE)
    nxt = None
    if get("llm.embed_next_model"):
        # A forgotten profile is not an error (nomic-v1 is the default), but for a model that needs the
        # other framing it silently builds vectors from unframed text.
        if not get("llm.embed_next_profile"):
            log.warning("LLM_EMBED_NEXT_MODEL=%s is set without LLM_EMBED_NEXT_PROFILE: using %s, which sends text "
                        "unframed. Set the profile if the model needs prompts (known: %s).",
                        get("llm.embed_next_model"), DEFAULT_PROFILE, ", ".join(PROFILES))
        nxt = Target(get("llm.embed_next_base_url").rstrip("/") or None, get("llm.embed_next_model"), get("llm.embed_next_profile") or DEFAULT_PROFILE)
        if nxt == active:
            nxt = None
    return {"active": active, "next": nxt, "previous": None}


class EmbeddingConfig:
    """Thread-safe active / next / previous targets, saved to ``state_file`` once changed at runtime."""

    def __init__(self, cfg, state_file: Optional[Path] = None):
        self._lock = threading.RLock()
        self.state_file = Path(state_file) if state_file else None
        self.version = 0
        self.damaged: Optional[str] = None
        self._held: Dict[str, Dict[str, Any]] = {}  # 'active' | 'next' -> {reason, transient, target}
        saved = self._read()
        self.source = "saved" if saved else "environment"
        self._state = saved or seed_from_cfg(cfg)

    def _read(self) -> Optional[Dict[str, Optional[Target]]]:
        if not self.state_file or not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                raise ValueError("unknown version")
            return {
                "active": normalize_target(data.get("active")),
                "next": normalize_target(data["next"]) if data.get("next") else None,
                "previous": normalize_target(data["previous"]) if data.get("previous") else None,
            }
        except Exception as exc:  # noqa: BLE001 — any damage: embeddings off, file kept for the operator
            self.damaged = f"The saved embedding configuration {self.state_file} cannot be read ({exc}). Fix or delete it."
            # Fail closed: an unknown profile refuses every embed call, and nothing is saved over the file.
            unavailable = Target(None, "unavailable", "unavailable")
            return {"active": unavailable, "next": None, "previous": None}

    # ---- seeded addresses refused at start-up (check_embedding_address); in memory, never saved ----

    def refuse_seeded(self, reasons: Dict[str, str], transient: bool = False) -> None:
        """An address the environment seeded is not on this network. A refused next is dropped; a
        refused active turns embedding off (unknown profile: every embed call refuses) and locks
        changes like a damaged file, because falling back to the chat engine would send it the same
        text. ``transient`` (the name did not resolve yet) can be undone by release_seeded."""
        with self._lock:
            if self.source != "environment":
                return
            for which in ("active", "next"):
                reason = reasons.get(which)
                if not reason:
                    continue
                if which in self._held:
                    self._held[which] = {**self._held[which], "reason": reason, "transient": transient}
                else:
                    self._held[which] = {"reason": reason, "transient": transient, "target": self._state[which]}
                    if which == "next":
                        self._state = {**self._state, "next": None}
                    else:
                        a = self._state["active"]
                        self._state = {**self._state, "active": Target(a.base_url, a.model, "unavailable")}
                if which == "active":
                    self.damaged = reason
            self.version += 1

    def release_seeded(self, which: str) -> None:
        """Takes back a transient refusal once the address resolves to this network."""
        with self._lock:
            held = self._held.get(which)
            if held is None or not held["transient"] or self.source != "environment":
                return
            del self._held[which]
            if which == "next":
                if self._state["next"] is None:  # not over an administrator's own choice
                    self._state = {**self._state, "next": held["target"]}
            else:
                self._state = {**self._state, "active": held["target"]}
                self.damaged = None
            self.version += 1

    def _persist(self, state) -> None:
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_name(f"{self.state_file.name}.{os.getpid()}.tmp")
        payload = {"version": 1, **{k: (v.to_json() if v else None) for k, v in state.items()}}
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.state_file)

    def _change(self, state) -> Dict[str, Any]:
        if self.damaged:
            raise EmbeddingConfigError(self.damaged, "badState", 409)
        self._persist(state)
        self._state = state
        self.version += 1
        self.source = "saved"
        return self.get()

    def get(self) -> Dict[str, Any]:
        with self._lock:
            s = self._state
            return {"active": s["active"].to_json(), "next": s["next"].to_json() if s["next"] else None,
                    "previous": s["previous"].to_json() if s["previous"] else None, "source": self.source,
                    "version": self.version, **({"damaged": self.damaged} if self.damaged else {})}

    def active(self) -> Target:
        with self._lock:
            return self._state["active"]

    def next(self) -> Optional[Target]:
        with self._lock:
            return self._state["next"]

    def previous(self) -> Optional[Target]:
        with self._lock:
            return self._state["previous"]

    def model_names(self) -> List[str]:
        with self._lock:
            return [t.model for t in self._state.values() if t]

    def set_next(self, raw) -> Dict[str, Any]:
        t = normalize_target(raw)
        with self._lock:
            if t == self._state["active"]:
                raise EmbeddingConfigError("That model, profile and address are already active.", "alreadyActive", 409)
            return self._change({**self._state, "next": t})

    def clear_next(self) -> Dict[str, Any]:
        with self._lock:
            return self._change({**self._state, "next": None})

    def switch_to_next(self, expected: Optional[Target] = None) -> Dict[str, Any]:
        """next becomes active; the old active is kept as previous. ``expected`` is what the caller
        checked readiness for: if next is something else by now, nothing switches."""
        with self._lock:
            if not self._state["next"]:
                raise EmbeddingConfigError("No re-index is under way.", "noNext", 409)
            if expected is not None and self._state["next"] != expected:
                raise EmbeddingConfigError("The re-index target changed while the switch was being checked. Nothing was switched.",
                                           "nextChanged", 409)
            return self._change({"active": self._state["next"], "next": None, "previous": self._state["active"]})

    def rollback(self) -> Dict[str, Any]:
        with self._lock:
            if not self._state["previous"]:
                raise EmbeddingConfigError("There is no previous embedding model to return to.", "noPrevious", 409)
            return self._change({"active": self._state["previous"], "next": None, "previous": self._state["active"]})

    def finish(self) -> Dict[str, Any]:
        with self._lock:
            if not self._state["previous"]:
                raise EmbeddingConfigError("There is no previous embedding model to remove.", "noPrevious", 409)
            return self._change({**self._state, "previous": None})


# ---- where an embeddings server may be (mirrors noevia-core's embedding-address.cjs) ----
#
# A re-index sends the Diary's text to that address, so it must be on this network: loopback, a
# private range, or a compose service name that resolves to one. The decision is positive: the URL
# must parse, the host must resolve (a literal is its own answer) and EVERY answer must be a local
# address. Public, unspecified, link-local, multicast and reserved addresses, and the IPv6 forms that
# carry another address, are refused. An unresolvable name is "transient" (it may resolve later); a
# public or malformed answer is permanent.

_V6_RESERVED = tuple(ipaddress.ip_network(n) for n in ("64:ff9b::/96", "2001::/32", "2001:db8::/32", "100::/64"))
_V4_RESERVED = tuple(ipaddress.ip_network(n) for n in ("0.0.0.0/8", "192.0.0.0/24", "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"))


@dataclass(frozen=True)
class AddressCheck:
    ok: bool
    reason: str = ""
    transient: bool = False


def non_local_reason(address: str) -> Optional[str]:
    """None when ``address`` is on this network (loopback, private, Tailscale), else why not."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return "not an address"
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_global:
        return "a public address"
    if ip.is_unspecified:
        return "the unspecified address"
    if ip.is_link_local:
        return "a link-local address"
    if ip.is_multicast:
        return "a multicast address"
    if ip.version == 4:
        if ip.packed[0] >= 240 or any(ip in n for n in _V4_RESERVED):
            return "a reserved address"
    else:
        if int(ip) >> 32 == 0 and int(ip) != 1:  # ::/96 minus ::1: v4-compatible forms
            return "a reserved address"
        if any(ip in n for n in _V6_RESERVED):
            return "a reserved address"
    return None


def check_embedding_address(url: str, resolver: Callable[[str], Sequence[str]] = None) -> AddressCheck:
    """Never raises. ``resolver(host)`` returns every address of the name (default: getaddrinfo)."""
    def resolve(host: str) -> List[str]:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return sorted({str(i[4][0]) for i in infos})
    resolver = resolver or resolve
    try:
        parts = urlsplit(str(url))
        host = parts.hostname or ""
        parts.port  # noqa: B018 — raises ValueError for a bad port
    except ValueError:
        return AddressCheck(False, "not a URL")
    if parts.scheme not in ("http", "https"):
        return AddressCheck(False, "not an http(s) URL")
    if parts.username or parts.password:
        return AddressCheck(False, "credentials in the URL")
    if not host:
        return AddressCheck(False, "no host")
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        try:
            addresses = list(resolver(host))
        except Exception as exc:  # noqa: BLE001 — a resolver that is down is not a public answer
            return AddressCheck(False, f"the name does not resolve ({type(exc).__name__})", transient=True)
    if not addresses:
        return AddressCheck(False, "the name does not resolve", transient=True)
    for address in addresses:
        why = non_local_reason(str(address))
        if why:
            return AddressCheck(False, f"{host} is not on this network ({address} is {why})")
    return AddressCheck(True)


def guard_seeded_addresses(config: "EmbeddingConfig", *, check: Callable[[str], AddressCheck] = check_embedding_address,
                           sleep: Callable[[float], None] = time.sleep, schedule: Optional[Callable[[Callable[[], None], float], None]] = None,
                           retry_delays: Sequence[float] = RETRY_DELAYS_S, recheck_s: float = RECHECK_S) -> None:
    """Start-up guard for LLM_EMBED_BASE_URL / LLM_EMBED_NEXT_BASE_URL. A refused value is logged and
    not used (in memory only); it never stops the service from starting. A name that does not
    resolve is retried with back-off, then held back and re-checked every ``recheck_s``, and used
    again as soon as it resolves to local addresses. Blocks until the first round has decided (run
    it on a thread); later re-checks run on daemon timers."""
    def default_schedule(fn: Callable[[], None], seconds: float) -> None:
        t = threading.Timer(seconds, fn)
        t.daemon = True
        t.start()
    schedule = schedule or default_schedule
    names = {"active": "LLM_EMBED_BASE_URL", "next": "LLM_EMBED_NEXT_BASE_URL"}
    try:
        state = config.get()
        if state["source"] != "environment":
            return
        urls = {w: state[w]["base_url"] for w in ("active", "next") if state[w] and state[w].get("base_url")}
        pending, results = dict(urls), {}  # results: the last answer for each address
        for delay in (0.0, *retry_delays):
            if delay:
                sleep(delay)
            for which in list(pending):
                results[which] = r = check(pending[which])
                if r.ok or not r.transient:
                    del pending[which]
            if not pending:
                break
    except Exception as exc:  # noqa: BLE001
        log.error("could not check the seeded embeddings address (%s)", type(exc).__name__)
        return

    def refuse_for_good(which: str, r: AddressCheck) -> None:
        log.error("%s %s refused: %s. The embeddings server must be on this network.", names[which], urls[which], r.reason)
        config.refuse_seeded({which: f"{names[which]} {urls[which]} refused: {r.reason}"})

    for which, r in results.items():
        if not r.ok and which not in pending:
            refuse_for_good(which, r)
    for which in pending:
        reason = results[which].reason
        log.warning("%s %s not used yet: %s. Checking again every %d s.", names[which], urls[which], reason, int(recheck_s))
        config.refuse_seeded({which: f"{names[which]} {urls[which]} not used yet: {reason}"}, transient=True)

        def again(which: str = which) -> None:
            if config.source != "environment":
                return  # an administrator has chosen since: the seed no longer matters
            try:
                r = check(urls[which])
                if r.ok:
                    config.release_seeded(which)
                    log.info("%s %s resolves to this network now; in use.", names[which], urls[which])
                    return
                if not r.transient:
                    refuse_for_good(which, r)
                    return
            except Exception:  # noqa: BLE001 — try again at the next round
                pass
            schedule(again, recheck_s)

        schedule(again, recheck_s)


class Embedder:
    """Embeds framed text for any target. Clients are cached per endpoint and closed with the embedder."""

    def __init__(self, cfg):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._clients: Dict[str, Any] = {}

    def _client(self, target: Target):
        from .llm import LLMClient  # local import: llm imports nothing from here

        key = target.base_url or ""
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                engine = str(self._cfg.get("llm.base_url") or "")
                client = LLMClient(
                    base_url=engine,
                    # The engine key goes to the engine only; a dedicated embeddings server never gets it.
                    api_key=(self._cfg.get("llm.api_key") or "") if not target.base_url else "",
                    embed_base_url=target.base_url or "",
                    timeout_s=float(self._cfg.get("llm.timeout_s", 300)),
                    max_retries=int(self._cfg.get("llm.max_retries", 3)),
                )
                self._clients[key] = client
            return client

    def embed(self, target: Target, texts: Sequence[str], kind: Union[str, Sequence[str]] = "document",
              titles: Optional[Sequence[Optional[str]]] = None) -> List[List[float]]:
        frame(target.profile, [], kind, titles)  # fail closed on an unknown profile before any call
        client = self._client(target)
        return embed_fitting(lambda ts, ks, tis: client.embed(frame(target.profile, ts, ks, tis), model=target.model),
                             texts, kind, titles)

    def close(self) -> None:
        with self._lock:
            clients, self._clients = list(self._clients.values()), {}
        for c in clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass


class EmbeddingView:
    """What a Retriever uses: the shared configuration, embedding through the state's own client
    for the target that client was built for (so its retries, stubs and timeouts apply as before),
    and through the shared embedder for any other target."""

    def __init__(self, config: EmbeddingConfig, embedder: Optional[Embedder], own_llm=None, own_target: Optional[Target] = None):
        self.config = config
        self.embedder = embedder
        self.own_llm = own_llm
        self.own_target = own_target

    def active(self) -> Target:
        return self.config.active()

    def embed(self, target: Target, texts: Sequence[str], kind: Union[str, Sequence[str]] = "document",
              titles: Optional[Sequence[Optional[str]]] = None) -> List[List[float]]:
        if self.own_llm is not None and (self.embedder is None or target == self.own_target):
            own = self.own_llm
            return embed_fitting(lambda ts, ks, tis: own.embed(frame(target.profile, ts, ks, tis), model=target.model),
                                 texts, kind, titles)
        return self.embedder.embed(target, texts, kind, titles)


# ---- shared per service: one configuration and embedder per state file ----

_shared_lock = threading.Lock()
_shared: Dict[str, "SharedEmbedding"] = {}


@dataclass
class SharedEmbedding:
    config: EmbeddingConfig
    embedder: Embedder


def state_path_for(cfg) -> Path:
    explicit = cfg.get("retrieval.embedding_state_path")
    if explicit:
        return Path(explicit)
    db = str(cfg.get("retrieval.db_path") or "")
    base = Path(db).parent if db and db != ":memory:" else Path(".")
    return base / "embedding-config.json"


def shared_for(cfg) -> SharedEmbedding:
    """The configuration and embedder of the service ``cfg`` belongs to (keyed by its state file)."""
    key = str(state_path_for(cfg).resolve())
    with _shared_lock:
        s = _shared.get(key)
        if s is None:
            s = SharedEmbedding(EmbeddingConfig(cfg, state_path_for(cfg)), Embedder(cfg))
            _shared[key] = s
        return s


def view_for(cfg, own_llm) -> EmbeddingView:
    shared = shared_for(cfg)
    own = seed_from_cfg(cfg)["active"]
    return EmbeddingView(shared.config, shared.embedder, own_llm=own_llm, own_target=own)


def reset_shared_for_tests() -> None:
    with _shared_lock:
        items = list(_shared.values())
        _shared.clear()
    for s in items:
        s.embedder.close()


__all__ = ["DEFAULT_PROFILE", "PROFILES", "frame", "clean_title", "Target", "normalize_target", "seed_from_cfg",
           "EmbeddingConfig", "EmbeddingConfigError", "Embedder", "EmbeddingView", "shared_for", "view_for",
           "state_path_for", "reset_shared_for_tests", "AddressCheck", "check_embedding_address", "non_local_reason",
           "guard_seeded_addresses"]
