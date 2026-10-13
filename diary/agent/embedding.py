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
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

log = logging.getLogger(__name__)

DEFAULT_PROFILE = "nomic-v1"
PROFILES = ("nomic-v1", "embeddinggemma-2")
KINDS = ("query", "document", "similarity")
_MODEL_RE = re.compile(r"^[^\s\x00-\x1f]{1,200}$")


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
        framed = frame(target.profile, texts, kind, titles)
        return self._client(target).embed(framed, model=target.model)

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
            return self.own_llm.embed(frame(target.profile, texts, kind, titles), model=target.model)
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
           "state_path_for", "reset_shared_for_tests"]
