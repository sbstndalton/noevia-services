"""JSON API for noevia's model management screens.

noevia's web server is the only client: it authenticates administrators and forwards
requests here over the internal Compose network. This module reuses the same service
functions as the original server-rendered pages, so both stay consistent while the pages
are phased out.

Every models.ini write is pinned to the file revision the client last read (sha256 of
the file) and fails with 409 when the file changed since, so an edit made by noevia's
own preset editor, a calibration run or an operator is never silently overwritten.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from . import autoconfig, db, gguf_meta, hw, ini, services, telemetry
from .config import settings
from .downloader import manager
from .utils import human_bytes

router = APIRouter(prefix="/api/v1")


def _plain(value: Any) -> Any:
    """JSON-safe copy of dataclasses, paths, sets and nested containers."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def revision() -> str:
    path = settings.models_ini_path
    data = path.read_bytes() if path.exists() else b""
    return hashlib.sha256(data).hexdigest()


def _require_revision(base: str | None) -> None:
    if not base:
        raise HTTPException(400, "baseRevision is required")
    if base != revision():
        raise HTTPException(409, "models.ini changed since you loaded it. Reload, review, then save again.")


def _field(f: ini.Field) -> dict:
    return {"key": f.key, "label": f.label, "kind": f.kind, "choices": list(f.choices),
            "placeholder": f.placeholder, "help": f.help}


def _schema() -> list[dict]:
    return [{"tier": label, "open": opened, "fields": [_field(f) for f in group]}
            for label, group, opened in ini.FORM_TIERS]


def _entry(g: services.GgufEntry, loaded: dict[str, list[str]]) -> dict:
    shape = services.model_shape(g.parts[0]) if g.parts and not g.is_companion else services.ModelShape()
    return {
        "key": g.route_key, "name": g.display_name, "stem": g.stem, "subdir": g.subdir,
        "bytes": g.total_bytes, "size": g.human_size, "modified": g.modified, "mtime": g.mtime,
        "sharded": g.is_sharded, "parts": len(g.parts), "companion": g.is_companion,
        "projector": {"name": g.companion_name, "bytes": g.companion_bytes} if g.companion_name else None,
        "sections": list(g.aliases), "modelId": g.model_id, "file": g.first_shard_rel,
        "shape": {"arch": shape.arch, "moe": shape.is_moe, "experts": shape.expert_count,
                  "active": shape.expert_used, "label": shape.label} if shape.known else None,
        "loadedOn": loaded.get(g.model_id, []),
        "fit": services.vram_fit_chips(g.total_bytes) if not g.is_companion else [],
    }


async def _loaded_map() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for b in await services.snapshot_llama_backends():
        for mid in [s.strip() for s in (b.loaded_model or "").split(",") if s.strip()]:
            out.setdefault(mid, []).append(b.name)
    return out


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.get("/overview")
async def overview() -> dict:
    snap = services.snapshot_models_dir()
    backends = await services.snapshot_llama_backends()
    return {
        "modelsDir": {"path": str(snap.path), "exists": snap.exists, "error": snap.error,
                      "disk": {"total": snap.disk.total, "free": snap.disk.free, "usedPct": snap.disk.used_pct,
                               "totalH": snap.disk.total_h, "freeH": snap.disk.free_h} if snap.disk else None},
        "models": sum(1 for g in snap.ggufs if not g.is_companion),
        "sections": len(ini.section_names()),
        "backends": [_plain(b) for b in backends],
        "activeDownloads": sum(1 for j in manager.snapshot() if j.status in ("queued", "downloading")),
        "revision": revision(),
    }


@router.get("/models")
async def models() -> dict:
    snap = services.snapshot_models_dir()
    loaded = await _loaded_map()
    return {"models": [_entry(g, loaded) for g in snap.ggufs if not g.is_companion],
            "unregistered": ini.unregistered_gguf_stems(), "revision": revision()}


def _find_entry(key: str) -> services.GgufEntry:
    if ".." in key or key.startswith("/") or key.count("/") > 1:
        raise HTTPException(400, "bad model key")
    snap = services.snapshot_models_dir()
    for g in snap.ggufs:
        if g.route_key == key or (not g.subdir and g.display_name == key):
            return g
    raise HTTPException(404, "model not found")


@router.get("/models/detail")
async def model_detail(key: str = Query(...)) -> dict:
    g = _find_entry(key)
    raw = gguf_meta.read_raw(g.parts[0])
    summary = gguf_meta.summarize(raw)
    summary.pop("chat_template", None)  # large; the features summary is what the UI shows
    return {**_entry(g, await _loaded_map()), "path": str(g.parts[0]), "summary": _plain(summary)}


@router.post("/models/delete")
def delete_models(body: dict = Body(...)) -> dict:
    results = []
    for item in body.get("models") or []:
        key = str(item)
        try:
            g = _find_entry(key)
        except HTTPException as e:
            results.append({"key": key, "ok": False, "message": e.detail}); continue
        ok, msg, freed = services.delete_gguf(g.display_name, g.subdir)
        results.append({"key": key, "ok": ok, "message": msg, "freed": freed, "freedH": human_bytes(freed) if freed else ""})
    return {"results": results}


# ---------- models.ini ----------

@router.get("/sections")
def sections() -> dict:
    return {"revision": revision(), "schema": _schema(),
            "sections": [{"name": s.name, "items": s.items, "hasFile": s.has_file,
                          "file": s.matched_file, "cli": s.cli} for s in ini.list_sections()],
            "unregistered": ini.unregistered_gguf_stems(), "backups": ini.list_backups()}


def _resolve_section_gguf(name: str) -> tuple[Path | None, str, str | None]:
    from .main import _resolve_section_gguf as resolve
    return resolve(name)


@router.get("/sections/{name}")
def section(name: str, defaults: bool = False) -> dict:
    if not ini.valid_section_name(name):
        raise HTTPException(400, "invalid section name")
    current = ini.get_section(name)
    hints: list[str] = []
    if defaults or current is None:
        from .main import _gguf_hints_for
        values, hints = _gguf_hints_for(name)
        extras = ""
    else:
        values, extras = ini.split_section_for_form(current)
    return {"name": name, "exists": current is not None, "values": values, "extras": extras,
            "hints": hints, "revision": revision(), "schema": _schema()}


@router.put("/sections/{name}")
def save_section(name: str, body: dict = Body(...)) -> dict:
    if not ini.valid_section_name(name):
        raise HTTPException(400, "invalid section name")
    _require_revision(body.get("baseRevision"))
    raw = body.get("values") or {}
    values: dict[str, str] = {}
    for f in ini.ALL_FIELDS:
        v = raw.get(f.key, "")
        if f.kind == "bool":
            values[f.key] = "true" if v in (True, "true", "on", "1") else ""
        else:
            values[f.key] = str(v if v is not None else "").strip()
    extras = str(body.get("extras") or "")
    # One section per save: a value or extra line may not open another section.
    if any("\n" in v or "\r" in v for v in values.values()):
        raise HTTPException(400, "values must be single lines")
    if any(line.strip().startswith("[") for line in extras.splitlines()):
        raise HTTPException(400, "extra lines cannot start a new section")
    ini.upsert_section(name, values, extras)
    return {"ok": True, "revision": revision(), "section": ini.get_section(name)}


@router.post("/sections/{name}/rename")
def rename_section(name: str, body: dict = Body(...)) -> dict:
    new = str(body.get("newName") or "").strip()
    if not ini.valid_section_name(new):
        raise HTTPException(400, "invalid new name")
    _require_revision(body.get("baseRevision"))
    if new in ini.section_names():
        raise HTTPException(409, "a section with that name already exists")
    if not ini.rename_section(name, new):
        raise HTTPException(404, "section not found")
    return {"ok": True, "revision": revision()}


@router.delete("/sections/{name}")
def delete_section(name: str, baseRevision: str = Query("")) -> dict:
    _require_revision(baseRevision)
    if not ini.delete_section(name):
        raise HTTPException(404, "section not found")
    return {"ok": True, "revision": revision()}


@router.get("/sections/{name}/autoconfig")
def section_autoconfig(name: str, preset: str = "", sessions: int = 1, spec: str = "") -> dict:
    from .main import _backend_list
    sessions = max(1, min(int(sessions or 1), 8))
    gguf_path, model_rel, rel = _resolve_section_gguf(name)
    if gguf_path is None or not gguf_path.is_file():
        return {"error": f"No model file found for '{name}'."}
    try:
        summary = gguf_meta.summarize(gguf_meta.read_raw(gguf_path))
    except (gguf_meta.GgufMetaError, OSError) as e:
        return {"error": f"Could not read the model file: {e}"}
    file_size = gguf_path.stat().st_size
    if rel and "/" in rel:
        try:
            file_size = sum(p.stat().st_size for p in gguf_path.parent.iterdir()
                            if p.is_file() and p.suffix.lower() == ".gguf" and not ini._is_companion(p.name))
        except OSError:
            pass
    try:
        telemetry.ingest(services._effective_container_names())
        measured = telemetry.stats_for(model_path=model_rel, alias=name)
        history = telemetry.config_history(model_path=model_rel, alias=name)
    except Exception:  # noqa: BLE001 - measurements are optional
        measured, history = telemetry.Stats(), []
    rec = autoconfig.analyze(summary=summary, file_size=file_size, backends=_backend_list(),
                             model_rel=model_rel, current_section=ini.get_section(name), preset=preset,
                             n_sessions=sessions, models_dir=settings.models_dir, section_name=name,
                             model_subdir=rel.split("/", 1)[0] if rel and "/" in rel else "", spec_profile=spec)
    return {"section": name, "arch": summary.get("arch"), "params": (summary.get("general") or {}).get("params"),
            "fileBytes": file_size, "model": model_rel or rel or f"{name}.gguf",
            "recommendation": _plain(rec), "measured": _plain(measured), "history": _plain(history),
            "revision": revision()}


# ---------- backends ----------

@router.get("/backends")
async def backends() -> dict:
    out = []
    for b in await services.snapshot_llama_backends():
        stats = hw.stats_for(b.name)
        points = hw.history_for(b.name)
        out.append({**_plain(b), "stats": _plain(stats), "history": _plain(points[-120:])})
    return {"backends": out}


@router.post("/backends/{name}/restart")
def restart_backend(name: str) -> dict:
    if name not in services._effective_container_names():
        raise HTTPException(404, "unknown backend")
    ok, message = services.restart_llama_backend(name)
    return {"ok": ok, "message": message}


@router.get("/backends/{name}/logs")
def backend_logs(name: str, q: str = "", level: str = "", tail: int = 400) -> dict:
    if name not in services._effective_container_names():
        raise HTTPException(404, "unknown backend")
    ok, text = services.container_logs(name, tail=max(20, min(int(tail or 400), 2000)))
    if not ok:
        return {"ok": False, "error": text, "lines": []}
    needle = q.lower()
    lines = []
    for line in text.splitlines():
        low = line.lower()
        if needle and needle not in low:
            continue
        if level == "error" and not any(t in low for t in ("err", "fatal")):
            continue
        if level == "warn" and not any(t in low for t in ("err", "warn", "wrn", "fatal")):
            continue
        lines.append(line)
    return {"ok": True, "lines": lines}


@router.post("/backends/{name}/test")
async def backend_test(name: str, body: dict = Body(...)) -> dict:
    if name not in services._effective_container_names():
        raise HTTPException(404, "unknown backend")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")
    return await services.test_prompt(name, prompt, max_tokens=max(1, min(int(body.get("maxTokens") or 256), 4096)))
