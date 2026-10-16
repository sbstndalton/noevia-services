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
    # #1318: the native parent owns both badge writers; badge reads (benchmark runs, models) stay here.
    if (os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_BADGES_CONFIRMED") == "1" and request.method in {"PUT", "DELETE"}
            and request.url.path == "/api/v1/badges"):
        return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent runs the Hugging Face update check and writes update_check itself.
    if (os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_CHECK_UPDATES_CONFIRMED") == "1" and request.method == "POST"
            and request.url.path == "/api/v1/models/check-updates"):
        return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent reads the benchmark overview and asks this app only for the
    # runner's live job state (the runner thread lives here). The job controls stay here.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_OVERVIEW_CONFIRMED") == "1":
        if request.method == "GET" and request.url.path == "/api/v1/benchmark":
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == BENCH_JOB_PATH:
            return await _bench_job_call(request)
    # #1318: the native parent restarts and tests backends; restart errors stay recorded here.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_BACKEND_CONTROL_CONFIRMED") == "1":
        if request.method == "POST" and _native_backend_control(request):
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == BACKEND_RESTART_ERROR_PATH:
            return await _backend_restart_error(request)
    # #1318: the native parent writes the Hugging Face token; it asks this app only for the token
    # test (which calls Hugging Face). GET /settings follows its own switch.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_WRITE_CONFIRMED") == "1":
        if request.method == "PUT" and request.url.path == "/api/v1/settings":
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == HF_TOKEN_TEST_PATH:
            return await _hf_token_test(request)
    # #1318: the native parent is the only models.ini writer; it asks this app only for the safe
    # defaults plan (read-only). Any other method or path shape stays here.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_SECTION_WRITES_CONFIRMED") == "1":
        if _native_section_write(request.method, request.url.path):
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == SAFE_DEFAULTS_PLAN_PATH:
            return await _safe_defaults_plan(request)
    # #1318: the native parent reads canonical benchmark runs itself and asks this app only for the
    # live chart values (capacity_gb, vram_predicted). Other id spellings stay here.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_BENCH_RUNS_CONFIRMED") == "1":
        if request.method == "GET" and _canonical_bench_run(request):
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == BENCH_RUN_RUNTIME_PATH:
            return await _bench_run_runtime(request)
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


# #1318: the parent's private call that keeps `GET /backends`' last_restart_error: PUT
# {"name": str, "error": str | null} sets or clears it. 404 unless the parent owns restarts.
BACKEND_RESTART_ERROR_PATH = "/api/v1/internal/backend-restart-error"
_BACKENDS_PREFIX = b"/api/v1/backends/"


def _native_backend_control(request: Request) -> bool:
    """`POST /api/v1/backends/{name}/restart|test` with a plain ASCII name (no `%`): the shapes the
    parent answers, judged on the raw path."""
    raw = request.scope.get("raw_path") or request.url.path.encode()
    if not raw.startswith(_BACKENDS_PREFIX):
        return False
    name, _, action = raw[len(_BACKENDS_PREFIX):].partition(b"/")
    return (action in (b"restart", b"test") and 0 < len(name) <= 256 and name.isascii()
            and b"%" not in name)


async def _backend_restart_error(request: Request):
    import json as _json
    if request.method != "PUT":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    try:
        body = _json.loads(await request.body())
        name, error = body["name"], body.get("error")
        if not isinstance(name, str) or not (error is None or isinstance(error, str)):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        return Response('{"detail":"invalid restart error"}', status_code=422, media_type="application/json")
    from . import services
    with services._restart_errors_lock:
        if error is None:
            services._restart_errors.pop(name, None)
        else:
            services._restart_errors[name] = error
    return Response('{"ok":true}', media_type="application/json")


# #1318: the parent's private call for `PUT /settings`'s token test: `put_settings`' own test of the
# stored token, `{"ok": bool, "message": str}`; 404 unless the native writer owns the PUT.
HF_TOKEN_TEST_PATH = "/api/v1/internal/hf-token-test"


async def _hf_token_test(request: Request):
    from fastapi.responses import JSONResponse
    if request.method != "GET":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    from . import hf
    try:
        ok, message = await hf.validate_token(db.get_setting("hf_token", ""))
    except Exception as e:  # noqa: BLE001 - the same answer put_settings gives
        ok, message = False, f"Test failed: {e}"
    return JSONResponse({"ok": ok, "message": message})


# #1318: the parent's private call for safe defaults (the values register_safe_defaults would
# write, or its refusal); 404 unless the native writer owns models.ini writes.
SAFE_DEFAULTS_PLAN_PATH = "/api/v1/internal/safe-defaults-plan"


def _native_section_write(method: str, path: str) -> bool:
    """The five models.ini writers, matched on the decoded path as FastAPI routes them."""
    if method == "PUT" and path == "/api/v1/models-ini":
        return True
    parts = path.split("/")
    if len(parts) < 5 or parts[:4] != ["", "api", "v1", "sections"] or not parts[4]:
        return False
    if len(parts) == 5:
        return method in {"PUT", "DELETE"}
    return len(parts) == 6 and method == "POST" and parts[5] in {"rename", "safe-defaults"}


async def _safe_defaults_plan(request: Request):
    """`{"status": 200, "values": {...}, "mtp": bool}`, or the refusal as
    `{"status": <code>, "detail": "..."}`. Nothing is written."""
    from fastapi import HTTPException
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse
    if request.method != "GET":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    name = request.query_params.get("name", "")

    def compute() -> dict:
        from . import api
        try:
            values, mtp = api._safe_defaults_plan(name)
        except HTTPException as e:
            return {"status": e.status_code, "detail": e.detail}
        return {"status": 200, "values": values, "mtp": mtp}
    return JSONResponse(await run_in_threadpool(compute))


# #1318: the parent's private call for the benchmark runner's live state (`job` in GET /benchmark);
# 404 unless the native parent owns that read. The parent never forwards this path.
BENCH_JOB_PATH = "/api/v1/internal/bench-job"


async def _bench_job_call(request: Request):
    """`api._bench_job()` exactly as GET /benchmark embeds it."""
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse
    if request.method != "GET":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    from . import api
    return JSONResponse(jsonable_encoder(api._bench_job()))


# #1318: the parent's private call for the benchmark run charts' live values; 404 unless the native
# parent owns the read. The parent never forwards this path.
BENCH_RUN_RUNTIME_PATH = "/api/v1/internal/bench-run-runtime"
_BENCH_RUN_PREFIX = b"/api/v1/benchmark/runs/"


def _canonical_bench_run(request: Request) -> bool:
    """The ids the native parent answers: `-?[0-9]+` inside i64, judged on the raw path so a
    percent-encoded id (which the parent forwards) is still served here."""
    raw = request.scope.get("raw_path") or request.url.path.encode()
    if not raw.startswith(_BENCH_RUN_PREFIX):
        return False
    seg = raw[len(_BENCH_RUN_PREFIX):]
    digits = seg[1:] if seg.startswith(b"-") else seg
    if not digits or not digits.isdigit() or not all(48 <= c <= 57 for c in digits):
        return False
    return -2 ** 63 <= int(seg) < 2 ** 63


async def _bench_run_runtime(request: Request):
    """`capacity_gb` and `vram_predicted` exactly as `helpers._bench_charts` computes them, for the
    aliases the parent names (its vram_labels, in order)."""
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse
    if request.method != "GET":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    aliases = request.query_params.getlist("alias")
    if len(aliases) > 1024:
        return Response('{"detail":"invalid runtime query"}', status_code=422, media_type="application/json")

    def compute() -> dict:
        from . import helpers
        capacity = 0.0
        try:
            for b in helpers._backend_list():
                capacity = max(capacity, float(b.get("vram_gb") or 0))
        except Exception:  # noqa: BLE001
            capacity = 0.0
        return {"capacity_gb": round(capacity, 2),
                "predicted": {a: helpers._predicted_vram_gb(a) for a in aliases}}
    return JSONResponse(await run_in_threadpool(compute))


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
