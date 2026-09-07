# Diary Companion service

Diary Companion is Cowork's optional, independently runnable FastAPI service. It provides a browser UI and an OpenAI-compatible `/v1` API, assembles diary context for conversations, and records selected exchanges into daily Markdown files.

## Storage

`CORPUS_BACKEND=local` is the default and uses `CORPUS_LOCAL_ROOT`. `CORPUS_BACKEND=webdav` uses `WEBDAV_BASE_URL`, `WEBDAV_USERNAME`, and `WEBDAV_PASSWORD`. `CORPUS_BACKEND=s3` uses any S3-compatible endpoint (MinIO, Backblaze B2, AWS S3, Garage, ...) via `S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, optional `S3_SESSION_TOKEN`, `S3_REGION`, and `S3_PREFIX`. `CORPUS_ROOT` adds an optional subdirectory within any backend.

All backends implement the same versioned storage contract. Local writes use locking, content hashes, `fsync`, and atomic replacement; WebDAV writes use ETag preconditions and bounded conflict retries; S3 writes require atomic conditional PUTs. Before its first write, the backend tests If-Match and If-None-Match on a unique disposable object under `.cowork-probes/` and refuses incompatible stores. Credentials need GetObject, PutObject, DeleteObject and ListBucket permissions for the configured prefix. The probe is removed afterward. SigV4 signing uses the Python standard library.

The default daily layout is `Entries/YYYY/Month/Month D, YYYY.md`; month views
aggregate those files in calendar order. Existing monthly corpora remain
supported with `DIARY_ENTRY_LAYOUT=monthly` and
`DIARY_MONTH_FILE_TEMPLATE={year}-{month02}.md`. Files under `AI Memory` and
`Raw Sources` are left untouched by the entry writer.

## Run locally

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn agent.app:app --reload --port 8010
```

Configuration defaults live in `config/config.yaml`. Environment variables override YAML values. Set an OpenAI-compatible endpoint and valid chat/embedding model IDs before starting a conversation.

## API and safety

- `GET /api/health`
- `POST /api/chat`
- `GET /api/day`
- `GET /api/months`
- `POST /api/entries/edit` — guarded correction of one logged exchange (see Editing below)
- `GET /v1/models`
- `POST /v1/chat/completions`

Set `DIARY_AUTH_TOKEN` before network exposure. The SQLite database contains the retrieval index and durable write journal and must live on persistent storage.

## Editing — the integrity guarantee

The logger keeps the user's own words near-verbatim; that guarantee extends to
corrections. Past entries are editable in the app, but only through
`POST /api/entries/edit`, which:

- targets **exactly one** logged exchange, identified by its hidden `xid` marker —
  nothing else in the file can be touched;
- routes through the same write-ahead journal and ETag-guarded write as every
  other corpus mutation (there is no separate, unguarded write path), so a crash
  before or after the underlying PUT replays idempotently;
- refreshes the retrieval index afterward, so a superseded chunk can never be
  served alongside the corrected text.

Editing is a human-initiated, visible action in the UI. The assistant never
rewrites the user's words on its own, and no pipeline step edits corpus text.

## Thoughtful diary conversations

Reflection is part of normal diary responses, not a separate screen or button.
Ask a question in the diary composer; the companion uses today's log, standing
sections, and retrieved older entries to respond in the same conversation.
The system prompt asks for specific, grounded observations when useful, and
brief acknowledgments for simple entries. It distinguishes facts from its own
interpretations. Assistant responses remain visibly separate from the user's
words; logging preserves that separation in the corpus.

Edits mark their document dirty in the durable SQLite journal before writing.
Dirty documents are excluded from retrieval until reindexing succeeds, including
after a crash. A failed correction reports that it is queued instead of claiming
it has been saved. Journal replay preserves operation ordering.

### Trust model: never expose this service directly

**This service MUST only ever be reachable from the Cowork web server's
container/proxy on an internal network. Do not publish its port, do not route
public traffic to it, and do not put it behind a shared reverse proxy that
forwards arbitrary requests.**

Tenant isolation here is a network-topology property, not a code property:
`DIARY_AUTH_TOKEN` is a single shared service token. Any holder of that token
can impersonate **any** tenant by supplying an arbitrary `X-Cowork-User-ID`
header — including permanently deleting that tenant's entire corpus via
`DELETE /api/internal/tenant`. The web server is what authenticates real users
and decides which tenant ID to forward. Because this service has no per-tenant
authentication of its own, direct exposure would let anyone with the token
act as anyone. This is a deliberate trade-off documented here as a hard
requirement; keep the service internal.
