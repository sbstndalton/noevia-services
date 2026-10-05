# Architecture

## Stack

- **FastAPI** — JSON routes under `/api/v1` (`app/api.py`), no async DB (sqlite is sync + fast enough). The upstream server-rendered pages (Jinja2, HTMX, Alpine, Tailwind) were removed in #806; noevia web is the only client.
- **docker SDK for Python** — container discovery, exec, stats.
- **sqlite** — app-level state (HF token, prompts, download history, avatar cache, benchmark results including each response's text, and your per-model capability badges).

Everything ships as one Docker image. Deps in `requirements.txt` are pinned. Python 3.12 slim base.

## File layout

```
app/
  main.py             FastAPI app: token middleware, startup hooks, mounts the /api/v1 router.
  api.py              The /api/v1 JSON routes (the contract in docs/spec-model-loader-api-v1.md).
  helpers.py          Helpers api.py shares (badges, section/GGUF resolution, backend list, bench charts).
  services.py         docker SDK wrapper and models-dir helpers.
  autoconfig.py       KV cache math, preset picker, VRAM fit, MoE offload strategy.
  hw.py               Background sampler thread, per-container stats cache.
  hf.py               HF Hub API client (search, tree, avatar cache).
  gguf_meta.py        Hand-rolled GGUF v3 metadata reader (no numpy).
  ini.py              models.ini schema (98 fields, 10 tiers) + parser + writer.
  downloader.py       Parallel-range download engine with sqlite-backed job history.
  db.py               Sqlite prefs, benchmark results, badges (schema inline in init()).
  migrate_layout.py   One-shot: flat GGUF layout -> per-model-subdir + absolute paths.
  config.py           pydantic-settings for env vars.
  utils.py            Small helpers (size formatting, stem parsing, etc.).
```

## Data flow

### On startup

1. `app/main.py` calls `hw.start_sampler()` — spawns a daemon thread that polls each discovered llama container every 2 seconds and caches stats.
2. Docker client connects to `/var/run/docker.sock`.
3. FastAPI serves JSON from cache-warm stats.

### On download

1. `POST /api/v1/downloads` inserts a row into `downloads` sqlite table with status=`queued`.
2. A background worker (started at first request) picks it up, splits the file into 8 byte-range chunks, opens 8 concurrent HTTPS streams.
3. Each chunk writes to a `.part-N` file. Progress + speed sample every 500ms into an in-memory ring buffer for the sparkline.
4. On completion, chunks are concatenated in order, temp files removed, `status=done`.
5. If a companion `*mmproj*.gguf` exists in the same repo, it's auto-queued into the same subdir.

### On config save

1. noevia web sends `PUT /api/v1/sections/{name}` (or the whole file through `PUT /api/v1/models-ini`).
2. `ini.py` reads current `models.ini`, updates the named section, writes atomically via `os.replace()`.
3. Before overwriting, current file is copied to `models.ini.bak-<timestamp>`. Only the 10 newest backups are kept.

### OpenWebUI

The upstream OpenWebUI reconciler (direct `webui.db` writes over `docker exec`) was removed in #806: noevia is the client, so there is no OpenWebUI to keep in sync.

## Design choices worth calling out

### One directory per model

`/models/<stem>/<file>.gguf` instead of flat `/models/<file>.gguf`. Reason: many models ship a matching `mmproj-*.gguf` (vision projector). In a flat layout, mmproj files from two different models will collide by filename. With per-stem subdirs, they can't.

`app/migrate_layout.py` does the one-shot conversion and rewrites `models.ini` with absolute paths.

### Explicit absolute `model = /models/<stem>/<file>.gguf` in every ini section

`llama-server --models-dir /models --models-preset foo.ini` does *not* join the `model` value with `--models-dir`. It takes the value literally. If you write `model = foo.gguf`, llama-server looks in `$PWD`, not `/models`. Autoconfig always writes absolute paths to avoid this footgun.

### Hand-rolled GGUF metadata parser (no numpy)

`gguf_meta.py` reads the GGUF v3 header + metadata KV pairs manually. Weight tensors are skipped. This avoids pulling numpy (~50 MB) into the image just to parse a few dozen integers. About 200 lines.

### Sampler runs in a thread, not asyncio

Docker SDK is sync. Running `nvidia-smi` via `container.exec_run()` is sync. Wrapping this in an async loop with `run_in_executor` bought us nothing — the thread does the same work with less ceremony. 2-second sample interval is plenty for a homelab dashboard.

### Autoconfig is a suggestion, not an action

The autoconfig route returns a recommendation, but doesn't apply it — you manually save the form, so you can pull it up for reference without anything reaching `models.ini`.

Once you do press Fill, though, it is **destructive within its declared domain**: `AUTOCONFIG_DOMAIN` lists every key Autoconfig has an opinion about, and Fill clears the ones it wants unset as well as writing the ones it has values for. That is deliberate — the alternative left stale placement keys behind that Save then wrote straight back — but it means a hand-tuned value inside the domain will be cleared. Keys outside the domain are never touched. See `docs/AUTOCONFIG.md`.

## What's intentionally not here

- **No auth.** Personal-use tool. Reverse-proxy it if you need auth.
- **No queue backend.** In-process asyncio queue for downloads. Rebooting model-loader mid-download means resuming from the last chunk (which is fine since chunks are byte-range).
- **No metrics export.** Prometheus/Grafana are out of scope. noevia web shows the stats.
- **No user prefs beyond a global HF token.** Single-user tool.
- **No frontend.** JSON API only; noevia web renders every screen.
- **No tests.** Correctness is enforced by shipping to production of one (the developer) and fixing what breaks. Do not adopt this policy on a team.
