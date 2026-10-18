from __future__ import annotations

import logging
import os
import re
import string

from fastapi import FastAPI, Request
from fastapi.responses import Response

from . import db, downloader, hw, ini, model_files
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


def _draft_heads_path(path: str) -> bool:
    parts = path.split("/")
    return len(parts) == 6 and parts[:4] == ["", "api", "v1", "sections"] and parts[4] != "" and parts[5] == "draft-heads"


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
    # #1318: the native parent owns the benchmark job slot (prompt runs, sweeps, the startup
    # cleanup of leftover sweep containers). It asks this app only for the per-request
    # observations that need the GPU sampler and the backend logs, which live here.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_RUNNER_CONFIRMED") == "1":
        if _native_benchmark_runner(request.method, request.url.path):
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == BENCH_OBSERVE_PATH:
            return await _bench_observe(request)
    # #1318: the native parent owns the download job table (start, draft-head download, list,
    # cancel, clear) and runs the transfers. A start is still planned here, on the parent's
    # private request: the same route runs with its validation and Hugging Face lookups, queues
    # nothing, and the files it would have queued go back to the parent with its answer.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_DOWNLOADS_CONFIRMED") == "1":
        if _native_downloads(request.method, request.url.path):
            if request.headers.get(DOWNLOAD_PLAN_HEADER) == "1" and _download_start(request.method, request.url.path):
                return await _download_plan(request, call_next)
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent reads GET /api/v1/sections/{name}/draft-heads (the section's model file,
    # its local and built-in heads, the download history and the Hugging Face listing). Other
    # methods keep FastAPI's 405 here; a name that decodes to a slash never matches the route.
    if (os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_DRAFT_HEADS_CONFIRMED") == "1" and request.method == "GET"
            and _draft_heads_path(request.url.path)):
        return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent owns model deletion (the container check and every file removal).
    # Other methods keep FastAPI's 405 here.
    if (os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_MODEL_DELETE_CONFIRMED") == "1" and request.method == "POST"
            and request.url.path == "/api/v1/models/delete"):
        return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
    # #1318: the native parent restarts and tests backends; restart errors stay recorded here.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_BACKEND_CONTROL_CONFIRMED") == "1":
        if request.method == "POST" and _native_backend_control(request):
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == BACKEND_RESTART_ERROR_PATH:
            return await _backend_restart_error(request)
    # #1318: the native parent answers GET /api/v1/sections/{name}/autoconfig (the section's file, its
    # header and the analysis) and asks this app only for the live inputs: the telemetry measurements
    # and the backend inventory. Other methods keep FastAPI's 405 here.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_SECTION_AUTOCONFIG_CONFIRMED") == "1":
        if request.method == "GET" and _section_autoconfig_path(request.url.path):
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == SECTION_AUTOCONFIG_RUNTIME_PATH:
            return await _section_autoconfig_runtime(request)
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
    # #1318: the native parent samples the backends and answers GET /api/v1/backends; it pushes its
    # samples here for this app's other readers and asks this app only for the GPU reads that
    # need `docker exec`. Other methods keep FastAPI's 405 here.
    if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_BACKENDS_CONFIRMED") == "1":
        if request.method == "GET" and request.url.path == "/api/v1/backends":
            return Response('{"detail":"Native model manager owns this endpoint"}', status_code=503, media_type="application/json")
        if request.url.path == BACKEND_STATS_PATH:
            return await _backend_stats(request)
        if request.url.path == GPU_EXEC_PROBE_PATH:
            return await _gpu_exec_probe(request)
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


# #1318: the parent's private calls while it samples the backends (404 unless it does).
# PUT {"backends": {name: BackendStats}} replaces this app's backend cache (what `hw.stats_for`
# answers to the fit chips, autoconfig and the benchmark observer); GET ?name=&probe= runs one
# GPU read that needs `docker exec` and answers {"gpu": GpuStats | null}.
BACKEND_STATS_PATH = "/api/v1/internal/backend-stats"
GPU_EXEC_PROBE_PATH = "/api/v1/internal/gpu-exec-probe"
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}")


async def _backend_stats(request: Request):
    import json as _json
    if request.method != "PUT":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    try:
        body = _json.loads(await request.body())
        if not isinstance(body, dict) or set(body) != {"backends"}:
            raise ValueError("body")
        hw.load_pushed(body["backends"])
    except (ValueError, TypeError, KeyError):
        return Response('{"detail":"invalid backend stats"}', status_code=422, media_type="application/json")
    return Response('{"ok":true}', media_type="application/json")


async def _gpu_exec_probe(request: Request):
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse
    from . import services
    from .api import _plain
    if request.method != "GET":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    name, probe = request.query_params.get("name", ""), request.query_params.get("probe", "")
    if not _CONTAINER_NAME.fullmatch(name) or probe not in ("nvidia", "amd", "amd-nvidia"):
        return Response('{"detail":"invalid probe query"}', status_code=422, media_type="application/json")
    # #1986: `docker exec` only into an active backend, and never by an all-hex name, which Docker
    # also resolves as an id prefix of a different container (as the native ScopedControl refuses).
    # The same 422 as any other invalid probe query.
    if all(c in string.hexdigits for c in name) or name not in await run_in_threadpool(services._effective_container_names):
        return Response('{"detail":"invalid probe query"}', status_code=422, media_type="application/json")
    gpu = await run_in_threadpool(hw.exec_probe, name, probe)
    return JSONResponse({"gpu": _plain(gpu)})


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


# #1318: the parent's private call for the live inputs of GET /api/v1/sections/{name}/autoconfig;
# 404 unless the native parent owns that read. The parent never forwards this path.
SECTION_AUTOCONFIG_RUNTIME_PATH = "/api/v1/internal/section-autoconfig-runtime"
_SECTION_AUTOCONFIG_RUNTIME_BODY = 256 * 1024


def _section_autoconfig_path(path: str) -> bool:
    """`/api/v1/sections/{name}/autoconfig` on the decoded path, as FastAPI routes it."""
    parts = path.split("/")
    return len(parts) == 6 and parts[:4] == ["", "api", "v1", "sections"] and parts[4] != "" and parts[5] == "autoconfig"


async def _section_autoconfig_runtime(request: Request):
    """POST {"section": name, "model": model_rel} -> {"measured": ..., "history": [...],
    "promptTps": "...", "backends": "[...]"}: what `api.section_autoconfig` reads from live state, in
    its order. First the telemetry ingest and the section's measurements (any failure there gives the
    empty ones, as in the route), then `helpers._backend_list()` (a failure there is this app's 500,
    as it is for the route). `measured` and `history` are rendered as the route's `-> dict` answer
    renders them (pydantic's JSON mode, then the JSON response class), so the parent places them
    verbatim. The analysis inputs keep every float, non-finite ones included: `promptTps` is
    `repr(float(measured.prompt_p50 or 0))` and `backends` is the list as JSON text (NaN and Infinity
    allowed), each carried as a string."""
    import json as _json

    from fastapi.concurrency import run_in_threadpool
    if request.method != "POST":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    raw = await request.body()
    try:
        if len(raw) > _SECTION_AUTOCONFIG_RUNTIME_BODY:
            raise ValueError("body")
        body = _json.loads(raw)
        if (not isinstance(body, dict) or set(body) != {"section", "model"}
                or not all(isinstance(body[k], str) for k in body)):
            raise ValueError("body")
    except ValueError:
        return Response('{"detail":"invalid runtime body"}', status_code=422, media_type="application/json")

    def compute() -> str:
        from pydantic import TypeAdapter

        from . import services, telemetry
        from .api import _plain
        from .helpers import _backend_list
        try:
            telemetry.ingest(services._effective_container_names())
            measured = telemetry.stats_for(model_path=body["model"], alias=body["section"])
            history = telemetry.config_history(model_path=body["model"], alias=body["section"])
        except Exception:  # noqa: BLE001 - measurements are optional, as in the route
            measured, history = telemetry.Stats(), []
        backends = _backend_list()
        shown = TypeAdapter(dict).dump_python({"measured": _plain(measured), "history": _plain(history)},
                                              mode="json")
        return _json.dumps({"measured": shown["measured"], "history": shown["history"],
                            "promptTps": repr(float(getattr(measured, "prompt_p50", 0) or 0)),
                            "backends": _json.dumps(backends, ensure_ascii=False, allow_nan=True,
                                                    separators=(",", ":"))},
                           ensure_ascii=False, separators=(",", ":"))
    return Response(await run_in_threadpool(compute), media_type="application/json")


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


# #1318: the four job routes the native benchmark runner answers.
_BENCHMARK_RUNNER_ROUTES = {("POST", "/api/v1/benchmark/start"), ("POST", "/api/v1/benchmark/sweep"),
                            ("GET", "/api/v1/benchmark/progress"), ("POST", "/api/v1/benchmark/cancel")}


def _native_benchmark_runner(method: str, path: str) -> bool:
    return (method, path) in _BENCHMARK_RUNNER_ROUTES


# #1318: the parent's private planning request carries this header (never forwarded from a
# client: the parent copies only the content type and the token).
DOWNLOAD_PLAN_HEADER = "x-noevia-download-plan"


def _download_start(method: str, path: str) -> bool:
    parts = path.split("/")
    return method == "POST" and (
        path == "/api/v1/downloads"
        or (len(parts) == 7 and parts[:4] == ["", "api", "v1", "sections"] and parts[4] != ""
            and parts[5:] == ["draft-heads", "download"]))


def _native_downloads(method: str, path: str) -> bool:
    parts = path.split("/")
    return (_download_start(method, path) or (method, path) in {("GET", "/api/v1/downloads"),
                                                                ("POST", "/api/v1/downloads/clear")}
            or (method == "POST" and len(parts) == 6 and parts[:4] == ["", "api", "v1", "downloads"]
                and parts[4] != "" and parts[5] == "cancel"))


async def _download_plan(request: Request, call_next):
    """Run the start route without queueing: `{"status", "body", "jobs"}` where status and body
    are the route's own answer (byte for byte) and jobs are the files it would have queued, in
    order, as `{"repo_id", "filename", "url", "total_bytes"}`. An exception in the route is this
    app's 500, as it would be for the public route."""
    from fastapi.responses import JSONResponse

    from . import downloader
    planned: list[dict] = []
    token = downloader.PLANNING.set(planned)
    try:
        response = await call_next(request)
    finally:
        downloader.PLANNING.reset(token)
    body = b"".join([chunk async for chunk in response.body_iterator])
    return JSONResponse({"status": response.status_code, "body": body.decode("utf-8"),
                         "jobs": planned if response.status_code == 200 else []})


# #1318: the parent's private call for one prompt-suite request's observations, which need this
# app's GPU sampler and backend log parser: `{"peak_vram_json": json.dumps(bench._peak_vram(b)),
# "contended": bench._other_traffic(b, aliases, since), "load_ms": bench._read_load_ms(b, alias)
# when cold, else null}`. 404 unless the native runner owns the job slot.
BENCH_OBSERVE_PATH = "/api/v1/internal/bench-observe"


async def _bench_observe(request: Request):
    import json as _json

    from fastapi.responses import JSONResponse
    if request.method != "GET":
        return Response('{"detail":"Method Not Allowed"}', status_code=405, media_type="application/json")
    q = request.query_params
    backend, alias = q.get("backend", ""), q.get("alias", "")
    try:
        since = float(q.get("since", ""))
    except ValueError:
        return Response('{"detail":"invalid since"}', status_code=422, media_type="application/json")
    cold = q.get("cold") == "1"
    if not backend:
        return Response('{"detail":"invalid backend"}', status_code=422, media_type="application/json")
    from . import bench
    peak = bench._peak_vram(backend)
    contended = bool(bench._other_traffic(backend, set(q.getlist("aliases")), since))
    load_ms = bench._read_load_ms(backend, alias) if cold else None
    return JSONResponse({"peak_vram_json": _json.dumps(peak), "contended": contended, "load_ms": load_ms})


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


@app.exception_handler(downloader.DeleteInProgress)
async def _delete_in_progress(_request: Request, exc: downloader.DeleteInProgress):
    """noevia#1972: a download start whose destination a running delete may remove."""
    from fastapi.responses import JSONResponse
    return JSONResponse({"detail": str(exc)}, status_code=409)


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
        # #1318: the native runner owns the job slot and cleans up leftover sweep containers
        # itself; two owners would race over the same containers and state.
        if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_RUNNER_CONFIRMED") != "1":
            bench.reap_orphans()
    except Exception as e:  # noqa: BLE001 - recovery is best effort; startup continues
        log.warning("benchmark orphan check failed: %s", type(e).__name__)
    # noevia#2010: under native section writes the parent is the only models.ini writer, and it
    # runs this migration itself, once, at its own startup under its write lock.
    if settings.migrate_cache_ram_on_start:
        if os.environ.get("NOEVIA_NATIVE_MODEL_MANAGER_SECTION_WRITES_CONFIRMED") == "1":
            log.info("cache-ram migration: left to the native models.ini writer")
        else:
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
