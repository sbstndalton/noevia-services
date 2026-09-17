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


def _entry(g: services.GgufEntry, loaded: dict[str, list[str]], badges: dict | None = None) -> dict:
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
        "badges": [_row(b) for b in badges.get(g.display_name, [])] if badges else [],
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
    from .main import _badges_for_files
    badges = _badges_for_files(snap)
    return {"models": [_entry(g, loaded, badges) for g in snap.ggufs if not g.is_companion],
            "unregistered": ini.unregistered_gguf_stems(), "revision": revision()}


def _find_entry(key: str) -> services.GgufEntry:
    if ".." in key or key.startswith("/") or key.count("/") > 4:
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
            "unregistered": ini.unregistered_gguf_stems(), "backups": ini.list_backups(), "raw": ini.raw_text()}


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


SAFE_DEFAULT_CTX = 8192


@router.post("/sections/{name}/safe-defaults")
def register_safe_defaults(name: str) -> dict:
    """Register a freshly downloaded GGUF with conservative settings, once.

    8k context (or less if the model is smaller), draft-mtp only when a draft head sits
    beside the model, `jinja` so the GGUF's own chat template is used, and no sampler keys,
    so the model's own sampling metadata stays in charge. Never overwrites: an existing
    section, or a file another section already points at, is left to the operator.
    """
    if not ini.valid_section_name(name):
        raise HTTPException(400, "invalid section name")
    if autoconfig._looks_like_draft(f"{name}.gguf") or "mmproj" in name.lower():
        raise HTTPException(400, "draft heads and projectors are not registered on their own")
    if ini.get_section(name) is not None:
        raise HTTPException(409, "settings already exist for this model")
    gguf_path, _model_rel, rel = _resolve_section_gguf(name)
    if gguf_path is None or rel is None:
        raise HTTPException(404, "no downloaded GGUF matches this name")
    if rel in ini._files_claimed_by_sections():
        raise HTTPException(409, "another settings section already uses this file")
    from .main import _gguf_hints_for
    values, _hints = _gguf_hints_for(name)
    if not values.get("model"):
        values["model"] = f"/models/{rel}"
    try:
        native = int(values.get("ctx-size") or 0)
    except ValueError:
        native = 0
    values["ctx-size"] = str(min(native, SAFE_DEFAULT_CTX) if native > 0 else SAFE_DEFAULT_CTX)
    subdir = rel.rsplit("/", 1)[0] if "/" in rel else ""
    head = autoconfig._find_mtp(settings.models_dir, name, subdir)
    if head:
        values["spec-type"] = "draft-mtp"
        values["spec-draft-model"] = head
    ini.upsert_section(name, values, "")
    return {"ok": True, "revision": revision(), "section": ini.get_section(name), "mtp": bool(head)}


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
def section_autoconfig(name: str, preset: str = "", sessions: int = 1, spec: str = "", vision: bool = True) -> dict:
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
                             model_subdir=rel.rsplit("/", 1)[0] if rel and "/" in rel else "", spec_profile=spec,
                             vision=vision)
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


# ---------- host, diagnosis ----------

def _row(r: Any) -> dict:
    return dict(r) if r is not None else {}


@router.get("/host")
def host() -> dict:
    points = hw.host_history()
    return {"current": _plain(points[-1]) if points else None, "history": _plain(points[-180:])}


def _logs_since_start(name: str) -> str:
    client = services._docker_client()
    if client is None:
        return ""
    try:
        c = client.containers.get(name)
        started = ((c.attrs or {}).get("State") or {}).get("StartedAt") or ""
        from datetime import datetime
        since = datetime.fromisoformat(started[:26].rstrip("Z").split(".")[0]) if started else None
        data = c.logs(since=since, tail=40000) if since else c.logs(tail=40000)
        return data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - diagnosis is best effort
        return ""


@router.get("/backends/{name}/diagnose")
def backend_diagnose(name: str) -> dict:
    if name not in services._effective_container_names():
        raise HTTPException(404, "unknown backend")
    from .diagnose import analyse
    return {"failures": _plain(analyse(_logs_since_start(name)))}


# ---------- Hugging Face settings ----------

@router.get("/settings")
def get_settings() -> dict:
    token = db.get_setting("hf_token", "")
    return {"hasToken": bool(token), "tokenHint": f"…{token[-4:]}" if len(token) > 8 else ""}


@router.put("/settings")
async def put_settings(body: dict = Body(...)) -> dict:
    from . import hf
    if "hfToken" in body:
        db.set_setting("hf_token", str(body.get("hfToken") or "").strip())
    result = get_settings()
    if body.get("test"):
        try:
            ok, message = await hf.validate_token(db.get_setting("hf_token", ""))
        except Exception as e:  # noqa: BLE001
            ok, message = False, f"Test failed: {e}"
        result["test"] = {"ok": ok, "message": message}
    return result


# ---------- search and downloads ----------

@router.get("/search")
async def search(q: str = "", sort: str = "downloads", limit: int = 30) -> dict:
    import httpx
    from . import hf
    from .main import _downloaded_and_still_present
    try:
        results = await hf.search_models(q.strip(), sort=sort, limit=max(1, min(int(limit or 30), 60)))
    except httpx.HTTPStatusError as e:
        return {"error": f"Hugging Face returned HTTP {e.response.status_code}", "results": []}
    except httpx.HTTPError as e:
        return {"error": f"Network error: {e}", "results": []}
    owners = [(m.id.split("/", 1)[0] if "/" in m.id else m.id) for m in results]
    avatars = await hf.owner_avatars(owners)
    have = _downloaded_and_still_present()
    return {"results": [{**_plain(m), "owner": o, "avatar": avatars.get(o, ""), "downloaded": have.get(m.id, [])}
                        for m, o in zip(results, owners)]}


@router.get("/search/repo")
async def search_repo(repo: str = Query(...)) -> dict:
    import httpx
    from . import hf
    from .main import _preset_estimates
    groups: list[dict] = []
    gated = ""
    try:
        detail = await hf.repo_detail(repo)
        by_base: dict[str, list] = {}
        for f in detail.files:
            by_base.setdefault(f.shard_base, []).append(f)
        for base, files in by_base.items():
            files.sort(key=lambda x: (x.shard_index or 0, x.path))
            total = sum(x.size for x in files)
            groups.append({"shardBase": base, "shards": files[0].shard_total if files[0].shard_index else None,
                           "bytes": total, "size": human_bytes(total),
                           "quant": next((x.quant for x in files if x.quant), None),
                           "fit": services.vram_fit_chips(total),
                           "projector": "mmproj" in files[0].path.lower(),
                           "files": [{"path": x.path, "bytes": x.size, "size": human_bytes(x.size), "quant": x.quant} for x in files]})
        groups.sort(key=lambda g: (0 if g["files"][0]["path"].lower().endswith(".gguf") else 1, g["projector"], g["shardBase"].lower()))
        probe = next((g for g in groups if g["files"][0]["path"].lower().endswith(".gguf") and not g["projector"]), None)
        mm = [g["bytes"] for g in groups if g["projector"]]
        if probe:
            summary = await hf.gguf_header(repo, probe["files"][0]["path"])
            gated = hf.gated_reason(repo) if not summary else ""
            if summary:
                for g in groups:
                    if g["files"][0]["path"].lower().endswith(".gguf") and not g["projector"]:
                        g["estimates"] = _preset_estimates(summary, g["bytes"], (min(mm) / 1024 ** 3) if mm else 0.0)
                        g["nativeCtx"] = (summary.get("model") or {}).get("context_length") or 0
    except httpx.HTTPStatusError as e:
        return {"repo": repo, "error": f"Hugging Face returned HTTP {e.response.status_code}", "groups": []}
    except httpx.HTTPError as e:
        return {"repo": repo, "error": f"Network error: {e}", "groups": []}
    return {"repo": repo, "groups": groups, "gated": gated}


def _job(j) -> dict:
    return {"id": j.id, "repo": j.repo_id, "filename": j.filename, "status": j.status, "error": j.error,
            "bytes": j.total_bytes, "downloaded": j.downloaded_bytes, "pct": round(j.pct, 1),
            "speed": j.speed_bps, "speedH": j.speed_h, "eta": j.eta_seconds, "etaH": j.eta_h,
            "startedAt": j.started_at, "completedAt": j.completed_at, "parallel": j.parallel,
            "history": j.speed_history_mbps[-60:],
            "chunks": [{"index": c.idx, "pct": round(c.pct, 1), "status": c.status, "speed": c.speed_bps} for c in j.chunks]}


@router.get("/downloads")
def downloads() -> dict:
    return {"jobs": [_job(j) for j in manager.snapshot()]}


@router.post("/downloads")
async def start_download(body: dict = Body(...)) -> dict:
    import httpx
    from . import hf
    from .main import _dest_for_companion, _dest_for_main, _model_stem
    repo, path, url = str(body.get("repo") or ""), str(body.get("path") or ""), str(body.get("url") or "").strip()
    queued: list[str] = []
    if url:
        if not url.startswith(("https://", "http://")):
            raise HTTPException(400, "URL must start with http:// or https://")
        from urllib.parse import unquote, urlparse
        name = str(body.get("filename") or "").strip() or unquote(urlparse(url).path.rsplit("/", 1)[-1])
        if not name or "/" in name or ".." in name:
            raise HTTPException(400, "Set a plain file name for this URL")
        total = 0
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                r = await client.head(url)
                total = int(r.headers.get("content-length") or 0) if r.status_code < 400 else 0
        except httpx.HTTPError:
            pass
        manager.enqueue_url(url=url, filename=f"{_model_stem(name)}/{name}", total_bytes=total)
        return {"queued": [name]}
    if not repo or "/" not in repo:
        raise HTTPException(400, "Choose a Hugging Face repository")
    detail = await hf.repo_detail(repo)
    shard_base = str(body.get("shardBase") or "")
    if shard_base:
        subdir = Path(shard_base).stem
        for f in detail.files:
            if f.shard_base == shard_base:
                manager.enqueue(repo_id=repo, hf_path=f.path, filename=f"{subdir}/{Path(f.path).name}", total_bytes=f.size)
                queued.append(f.path)
        main_stem = subdir
    else:
        match = next((f for f in detail.files if f.path == path), None)
        if match is None:
            raise HTTPException(404, "That file is not in the repository")
        base = Path(path).name
        if "mmproj" in base.lower():
            manager.enqueue(repo_id=repo, hf_path=path, filename=f"{_model_stem(base)}/{base}", total_bytes=match.size)
            return {"queued": [path]}
        main_stem, filename = _dest_for_main(base)
        manager.enqueue(repo_id=repo, hf_path=path, filename=filename, total_bytes=match.size)
        queued.append(path)
    # Companions from the same repo: its vision projector(s) and the smallest draft head.
    for f in [f for f in detail.files if "mmproj" in Path(f.path).name.lower()][:4]:
        manager.enqueue(repo_id=repo, hf_path=f.path, filename=_dest_for_companion(main_stem, Path(f.path).name), total_bytes=f.size)
        queued.append(f.path)
    heads = [f for f in detail.files if f.path.lower().endswith(".gguf") and "mmproj" not in Path(f.path).name.lower()
             and autoconfig._looks_like_draft(Path(f.path).name)]
    if heads and not shard_base:
        head = min(heads, key=lambda f: f.size or 0)
        manager.enqueue(repo_id=repo, hf_path=head.path, filename=_dest_for_companion(main_stem, Path(head.path).name), total_bytes=head.size)
        queued.append(head.path)
    return {"queued": queued}


@router.post("/downloads/{job_id}/cancel")
def cancel_download(job_id: str) -> dict:
    return {"ok": manager.cancel(job_id)}


@router.post("/downloads/clear")
def clear_downloads() -> dict:
    return {"cleared": manager.clear_finished()}


@router.post("/models/check-updates")
async def check_updates() -> dict:
    from . import hf
    from .main import _update_status_map
    records = db.download_records()
    if records:
        await hf.check_updates_for(records)
    return {"checked": len(records), "status": _update_status_map(services.snapshot_models_dir())}


@router.get("/models/updates")
def update_status() -> dict:
    from .main import _update_status_map
    return {"status": _update_status_map(services.snapshot_models_dir())}


# ---------- prompts ----------

@router.get("/prompts")
def prompts() -> dict:
    return {"prompts": [dict(p) for p in db.list_prompts()]}


@router.post("/prompts")
def add_prompt(body: dict = Body(...)) -> dict:
    name, text = str(body.get("name") or "").strip(), str(body.get("body") or "").strip()
    if not name or not text:
        raise HTTPException(400, "A name and prompt text are required")
    return {"id": db.add_prompt(name, text), "prompts": [dict(p) for p in db.list_prompts()]}


@router.delete("/prompts/{pid}")
def delete_prompt(pid: int) -> dict:
    if not db.delete_prompt(pid):
        raise HTTPException(404, "prompt not found")
    return {"prompts": [dict(p) for p in db.list_prompts()]}


# ---------- benchmarks and ratings ----------

def _bench_job() -> dict:
    from . import bench
    j = bench.state()
    return {**_plain(j), "pct": j.pct, "elapsed": j.elapsed_s, "eta": j.eta_s, "active": j.active}


@router.get("/benchmark")
def benchmark_overview() -> dict:
    from . import bench
    names = ini.section_names()
    return {"sections": names, "sweepArgs": {n: " ".join(bench.sweep_args_for_section(n)) for n in names},
            "prompts": [dict(p) for p in db.list_prompts()], "backends": services._effective_container_names(),
            "maxTokensDefault": bench.DEFAULT_MAX_TOKENS, "maxTokensCeiling": bench.MAX_TOKENS_CEILING,
            "job": _bench_job(), "runs": [_row(r) for r in db.bench_runs(limit=25)],
            "categories": [{"key": k, "label": v} for k, v in db.BADGE_CATEGORIES]}


@router.post("/benchmark/start")
def benchmark_start(body: dict = Body(...)) -> dict:
    from . import bench
    ok, err = bench.start(backend=str(body.get("backend") or ""), aliases=[str(a) for a in body.get("aliases") or []],
                          prompt_ids=[int(p) for p in body.get("promptIds") or []], reps=int(body.get("reps") or 3),
                          max_tokens=int(body.get("maxTokens") or bench.DEFAULT_MAX_TOKENS))
    if not ok:
        raise HTTPException(409, err or "The benchmark could not start")
    return {"job": _bench_job()}


@router.post("/benchmark/sweep")
def benchmark_sweep(body: dict = Body(...)) -> dict:
    from . import bench
    ok, err = bench.start_sweep(backend=str(body.get("backend") or ""), aliases=[str(a) for a in body.get("aliases") or []],
                                n_prompt=int(body.get("nPrompt") or 512), n_gen=int(body.get("nGen") or 128),
                                depths=str(body.get("depths") or "0,4096,16384"), reps=int(body.get("reps") or 3))
    if not ok:
        raise HTTPException(409, err or "The sweep could not start")
    return {"job": _bench_job()}


@router.get("/benchmark/progress")
def benchmark_progress() -> dict:
    return {"job": _bench_job()}


@router.post("/benchmark/cancel")
def benchmark_cancel() -> dict:
    from . import bench
    bench.cancel()
    return {"job": _bench_job()}


@router.get("/benchmark/runs/{run_id}")
def benchmark_run(run_id: int) -> dict:
    from .main import _bench_charts
    run = db.bench_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    results, sweeps, variants = db.bench_results(run_id), db.bench_sweeps(run_id), db.bench_variants(run_id)
    return {"run": _row(run), "variants": [_row(v) for v in variants], "results": [_row(r) for r in results],
            "sweeps": [_row(w) for w in sweeps], "charts": _bench_charts(run_id, results, sweeps),
            "badges": {v["alias"]: [_row(b) for b in db.badges_for(v["alias"])] for v in variants}}


@router.put("/badges")
def set_badge(body: dict = Body(...)) -> dict:
    alias = str(body.get("alias") or "")
    ok, err = db.badge_set(alias, str(body.get("category") or ""), int(body.get("rating") or 0),
                           str(body.get("note") or ""), int(body.get("runId") or 0) or None)
    if not ok:
        raise HTTPException(400, err)
    return {"badges": [_row(b) for b in db.badges_for(alias)]}


@router.delete("/badges")
def clear_badge(alias: str = Query(...), category: str = Query(...)) -> dict:
    db.badge_clear(alias, category)
    return {"badges": [_row(b) for b in db.badges_for(alias)]}
