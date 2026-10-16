from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import Response

from . import db, hw, ini, model_files
from .config import settings

log = logging.getLogger(__name__)

app = FastAPI(title="Model Loader")

# noevia#1071: the native parent (noevia-rs bins/noevia-model-manager) always owns these; this app
# has no handler for them and refuses them unconditionally (the API contract still lists them).
NATIVE_OWNED_ROUTES = frozenset({("GET", "/api/v1/health"), ("GET", "/api/v1/backends/{name}/logs")})
_NATIVE_OWNED = '{"detail":"Native model manager owns this endpoint"}'


def _native_owned(path: str) -> bool:
    parts = path.split("/")
    return path == "/api/v1/health" or (
        len(parts) == 6 and parts[:4] == ["", "api", "v1", "backends"] and parts[4] and parts[5] == "logs")


@app.middleware("http")
async def _require_token(request: Request, call_next):
    import hmac
    token = settings.model_loader_token
    if token and request.url.path != "/api/v1/health":
        sent = request.headers.get("x-model-loader-token", "")
        if not hmac.compare_digest(sent.encode(), token.encode()):
            return Response('{"error":"unauthorized"}', status_code=401, media_type="application/json")
    # This app only ever runs as the native parent's supervised loopback child (noevia#1071), so
    # health and logs are refused whatever the environment says, even if the Rust handler fails;
    # never silently serve them twice.
    if _native_owned(request.url.path):
        return Response(_NATIVE_OWNED, status_code=503, media_type="application/json")
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED") == "1" and request.url.path == "/api/v1/host":
        return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_DIAGNOSE_CONFIRMED") == "1" and request.method == "GET":
        parts = request.url.path.split("/")
        if len(parts) == 6 and parts[:4] == ["", "api", "v1", "backends"] and parts[4] and parts[5] == "diagnose":
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED") == "1" and request.method == "GET":
        path = request.url.path
        parts = path.split("/")
        if path == "/api/v1/sections" or (len(parts) == 5 and parts[:4] == ["", "api", "v1", "sections"] and parts[4]):
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED") == "1":
        path = request.url.path
        parts = path.split("/")
        if (path == "/api/v1/prompts" and request.method in {"GET", "POST"}) or (request.method == "DELETE" and len(parts) == 5 and parts[:4] == ["", "api", "v1", "prompts"] and parts[4]):
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent owns only the read; PUT (the single writer) stays here.
    if (os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_CONFIRMED") == "1" and request.method == "GET"
            and request.url.path == "/api/v1/settings"):
        return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent owns only this read; every update_check writer stays here.
    if (os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_MODEL_UPDATES_CONFIRMED") == "1" and request.method == "GET"
            and request.url.path == "/api/v1/models/updates"):
        return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent owns only this read; POST /api/v1/downloads (which checks its target
    # against the same list) stays here.
    if (os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_DOWNLOAD_TARGETS_CONFIRMED") == "1" and request.method == "GET"
            and request.url.path == "/api/v1/download-targets"):
        return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent owns this read and asks this app only for its live fields.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_MODEL_DETAIL_CONFIRMED") == "1":
        if request.method == "GET" and request.url.path == "/api/v1/models/detail":
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == MODEL_DETAIL_RUNTIME_PATH:
            return await _model_detail_runtime(request)
    return await call_next(request)


# #1318: the parent's private call, never a public route (so not in the API contract, and 404 when
# the native parent does not own the detail read). The native parent never forwards this path.
MODEL_DETAIL_RUNTIME_PATH = "/api/v1/internal/model-detail-runtime"


async def _model_detail_runtime(request: Request):
    """The live fields of GET /api/v1/models/detail, exactly as `api._entry` places them:
    `loadedOn` is `loaded[modelId]` (Docker discovery plus llama-server probes) and `fit` is
    `vram_fit_chips(bytes)` for a non-companion (the GPU sampler and host RAM). The parent reads
    the files itself and splices these in; the chips text is kept as rendered here."""
    from fastapi.responses import JSONResponse
    if request.method != "GET":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    size, fit = request.query_params.get("bytes", ""), request.query_params.get("fit", "")
    # #1889: at most 39 digits (enough for any 128-bit total the parent sends), so a hostile value never becomes a huge int.
    if not (size.isascii() and size.isdigit() and len(size) <= 39) or fit not in ("0", "1"):
        return Response('{"detail":"invalid runtime query"}', status_code=422, media_type="application/json")
    from . import api, services
    loaded = await api._loaded_map()
    return JSONResponse({"loaded": loaded, "fit": services.vram_fit_chips(int(size)) if fit == "1" else []})


from . import api as _api  # noqa: E402
app.include_router(_api.router)


@app.exception_handler(ini.DuplicateSectionsError)
async def _duplicate_sections(_request: Request, exc: ini.DuplicateSectionsError):
    """A write hit a models.ini with repeated [sections]; nothing was changed."""
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": f"{exc}. Repair the file (remove the repeated section) before saving."}, status_code=409)


@app.exception_handler(model_files.ModelFilesError)
async def _model_files_refused(_request: Request, exc: model_files.ModelFilesError):
    """model-files could not check a repository listing; nothing was served from it."""
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": "The repository file list could not be checked, so it was not used. "
                                   "See the model manager log."}, status_code=502)


@app.on_event("startup")
def _startup() -> None:
    # noevia#1071: model-files is the only listing/backup implementation; refuse to start without it.
    model_files.require_binary()
    db.init()
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED") != "1":
        db.seed_bench_prompts()
    hw.start_sampler()
    try:
        from . import bench
        bench.reap_orphans()
    except Exception as e:  # noqa: BLE001 - recovery is best effort; startup continues
        log.warning("benchmark orphan check failed: %s", type(e).__name__)
    if settings.migrate_cache_ram_on_start:
        _migrate_cache_ram()


def _migrate_cache_ram() -> None:
    """#697: presets without an explicit prompt cache would run on llama-server's 8 GiB default,
    and noevia's estimates would add that. Written once through the backed-up writer; the engine
    reads it on its next reload or restart. A failure is logged, never fatal."""
    try:
        changed = ini.migrate_cache_ram()
    except Exception as e:  # noqa: BLE001 - a malformed file must not stop the service
        log.warning("cache-ram migration skipped: %s", type(e).__name__)
        return
    if changed:
        log.warning("cache-ram migration: set an explicit bounded cache-ram in %d section(s): %s", len(changed), ", ".join(changed))
