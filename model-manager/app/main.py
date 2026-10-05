from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import Response

from . import db, hw, ini
from .config import settings

log = logging.getLogger(__name__)

app = FastAPI(title="Model Loader")


@app.middleware("http")
async def _require_token(request: Request, call_next):
    import hmac
    token = settings.model_loader_token
    if token and request.url.path != "/api/v1/health":
        sent = request.headers.get("x-model-loader-token", "")
        if not hmac.compare_digest(sent.encode(), token.encode()):
            return Response('{"error":"unauthorized"}', status_code=401, media_type="application/json")
    return await call_next(request)


from . import api as _api  # noqa: E402
app.include_router(_api.router)


@app.exception_handler(ini.DuplicateSectionsError)
async def _duplicate_sections(_request: Request, exc: ini.DuplicateSectionsError):
    """A write hit a models.ini with repeated [sections]; nothing was changed."""
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": f"{exc}. Repair the file (remove the repeated section) before saving."}, status_code=409)


@app.on_event("startup")
def _startup() -> None:
    db.init()
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
