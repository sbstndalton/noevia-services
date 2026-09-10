"""Configuration loader with environment-variable overrides.

Env overrides:
  LLM_BASE_URL, LLM_API_KEY, LLM_CHAT_MODEL, LLM_EMBED_MODEL,
  LLM_AUX_BASE_URL, LLM_AUX_MODEL, LLM_AUX_API_KEY,
  CORPUS_BACKEND, CORPUS_ROOT, CORPUS_LOCAL_ROOT,
  WEBDAV_BASE_URL, WEBDAV_USERNAME, WEBDAV_PASSWORD, CORPUS_REMOTE_ROOT (legacy),
  S3_ENDPOINT_URL, S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY,
  S3_SESSION_TOKEN, S3_REGION, S3_PREFIX,
  DIARY_MONTH_FILE_TEMPLATE, DIARY_INDEX_ENABLED,
  DIARY_ENTRY_LAYOUT, DIARY_ENTRIES_PREFIX,
  DB_PATH, DIARY_PORT
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)


class Config:
    """Nested attribute access over the YAML config plus env overrides."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        # deepcopy probes special attributes before _data is installed on the
        # reconstructed object. Avoid recursively invoking __getattr__ then.
        try:
            data = object.__getattribute__(self, "_data")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            value = data[name]
        except KeyError as exc:  # pragma: no cover - config contract
            raise AttributeError(f"missing config key: {name}") from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def as_dict(self) -> Dict[str, Any]:
        return self._data


def _apply_env(cfg: Config) -> None:
    env = os.environ

    def setpath(dotted: str, value: str) -> None:
        parts = dotted.split(".")
        node = cfg._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    mapping = {
        "LLM_BASE_URL": "llm.base_url",
        "LLM_API_KEY": "llm.api_key",
        "LLM_CHAT_MODEL": "llm.chat_model",
        "LLM_EMBED_MODEL": "llm.embed_model",
        "LLM_AUX_BASE_URL": "llm.aux.base_url",
        "LLM_AUX_MODEL": "llm.aux.model",
        "LLM_AUX_API_KEY": "llm.aux.api_key",
        "CORPUS_BACKEND": "corpus.backend",
        "CORPUS_ROOT": "corpus.root",
        "CORPUS_LOCAL_ROOT": "corpus.local.root",
        "WEBDAV_BASE_URL": "corpus.webdav.base_url",
        "WEBDAV_USERNAME": "corpus.webdav.username",
        "WEBDAV_PASSWORD": "corpus.webdav.password",
        "S3_ENDPOINT_URL": "corpus.s3.endpoint_url",
        "S3_BUCKET": "corpus.s3.bucket",
        "S3_ACCESS_KEY_ID": "corpus.s3.access_key",
        "S3_SECRET_ACCESS_KEY": "corpus.s3.secret_key",
        "S3_SESSION_TOKEN": "corpus.s3.session_token",
        "S3_REGION": "corpus.s3.region",
        "S3_PREFIX": "corpus.s3.prefix",
        "CORPUS_REMOTE_ROOT": "corpus.root",
        "DIARY_MONTH_FILE_TEMPLATE": "corpus.month_file_template",
        "DIARY_ENTRY_LAYOUT": "corpus.entry_layout",
        "DIARY_ENTRIES_PREFIX": "corpus.entries_prefix",
        "DIARY_INDEX_ENABLED": "corpus.index_enabled",
        "DB_PATH": "retrieval.db_path",
        "DIARY_PORT": "ui.port",
        "DIARY_AUTH_TOKEN": "ui.auth_token",
    }
    for env_key, dotted in mapping.items():
        value = env.get(env_key)
        if value is not None:
            if env_key == "DIARY_PORT":
                value = int(value)
            if env_key == "DIARY_INDEX_ENABLED":
                value = value.strip().lower() not in ("0", "false", "no", "off", "")
            setpath(dotted, value)
    if "CORPUS_REMOTE_ROOT" in env and "CORPUS_ROOT" not in env:
        log.warning("CORPUS_REMOTE_ROOT is deprecated; use CORPUS_ROOT")


# "default" is the placeholder every compose file falls back to when the operator
# has not chosen a model. It is not a model any backend actually serves, so a
# deployment that never set DIARY_AUX_MODEL sent the literal string "default" as
# the model name and every aux call 404d. Nothing broke loudly: the summariser
# falls back to logging the verbatim reply, so entries kept being written and
# only silently lost their topic headers. That went unnoticed for days.
#
# The aux model exists to be a cheaper model than the chat one, not a different
# capability — so the chat model is always a safe substitute. This mirrors what
# compose.yaml already does for llm.aux.base_url and api_key, which fall back to
# their main-endpoint counterparts; the model was simply never given the same
# treatment.
_PLACEHOLDER_MODEL = "default"


def _resolve_aux_model(cfg: Config) -> None:
    aux = (cfg.get("llm.aux.model") or "").strip()
    chat = (cfg.get("llm.chat_model") or "").strip()
    if aux and aux != _PLACEHOLDER_MODEL:
        return
    if not chat or chat == _PLACEHOLDER_MODEL:
        # Both unset: leave it alone rather than inventing a name. The operator
        # has configured nothing, and a 404 naming "default" is at least a
        # legible symptom of that.
        return
    parts = "llm.aux.model".split(".")
    node = cfg._data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = chat
    log.warning(
        "DIARY_AUX_MODEL is unset (or %r); falling back to the chat model %r. "
        "Set DIARY_AUX_MODEL to a smaller model the endpoint serves to make "
        "summarising and skip-classification cheaper.",
        _PLACEHOLDER_MODEL, chat,
    )


def load_config(path: Optional[str] = None) -> Config:
    if path is None:
        path = os.environ.get("DIARY_CONFIG", str(Path(__file__).resolve().parent.parent / "config" / "config.yaml"))
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:  # container image ships defaults; a mounted config is expected but optional
        data = {}
    cfg = Config(data)
    _apply_env(cfg)
    _resolve_aux_model(cfg)
    return cfg
