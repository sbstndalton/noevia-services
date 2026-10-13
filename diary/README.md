# Diary Companion service

Diary Companion is Cowork's optional, independently runnable FastAPI service. It has no UI of its own (noevia web is its only client). It exposes `/api/*` storage and entry endpoints plus an OpenAI-compatible `POST /v1/chat/completions`, assembles diary context for conversations, and records selected exchanges into daily Markdown files.

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

To run the tests, install the dev requirements (pytest, moto; not part of the Docker image) instead:

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

Configuration defaults live in `config/config.yaml`. Environment variables override YAML values. Set an OpenAI-compatible endpoint and valid chat/embedding model IDs before starting a conversation.

## Which model answers

The chat and aux model ids (`LLM_CHAT_MODEL`, `LLM_AUX_MODEL`) must be ids the engine serves. The
companion checks this at call time against the engine's `/v1/models` (remembered for 60 s): a
served configured id is used; otherwise it uses the first loaded chat model (else the first listed
one; sections launched with an embedding or reranking flag are never chosen), logs the choice once
at WARNING with ids only, and never leaves the configured endpoint. An engine that cannot be reached
is a logged `EngineUnreachable` error. `GET /api/health` reports `model` (the id in use),
`model_configured` and `model_source` (`configured`, `fallback`, `unverified`, `unreachable` or
`none_served`). The companion has no access to noevia's per-user Auto roles (they live in the web
server's workspace), so the fallback does not consult them.

## Embedding model and re-indexing (#1636)

Retrieval embeds each `###` subsection with the configured embedding model. `LLM_EMBED_MODEL` and
`LLM_EMBED_BASE_URL` name it (empty base URL: the chat engine); `LLM_EMBED_PROFILE` says how text is
framed for it: `nomic-v1` (default, text sent unchanged, as always) or `embeddinggemma-2` (the model
card's prompts: `task: search result | query: …` for searches, `title: none | text: …` for stored
subsections). Vectors from different models are kept apart in the same index file
(`agent/vector_slots.py`) and never compared.

Changing the model is a re-index, never an edit of the settings alone (`agent/embedding_migration.py`):
`POST /api/internal/embedding` (service token only, no tenant header; web's `/api/embedding` calls it)
with `set-next` and a target builds the new model's vectors for every tenant's index in the
background, from the text already in each index, while search keeps using the current model;
`switch` makes it active once it covers at least what the current model covers; `rollback` returns to
the previous model, whose vectors are kept until `finish`. `LLM_EMBED_NEXT_MODEL`,
`LLM_EMBED_NEXT_BASE_URL` and `LLM_EMBED_NEXT_PROFILE` start a re-index from the environment.
An embeddings address from the environment (`LLM_EMBED_BASE_URL`, `LLM_EMBED_NEXT_BASE_URL`) must be on this
network: it is checked at start-up and a public one is logged and not used. A name that does not resolve is
retried for about 30 s, then held back and re-checked every minute.
`EMBEDDING_REINDEX_DUTY` (default 0.5) caps how much of the time the background pass keeps the
embeddings server busy. Runtime changes are saved to `embedding-config.json` beside the service index
and win over the environment until that file is deleted.

## API and safety

- `GET /api/health`
- `GET /api/day`
- `GET /api/months`
- `POST /api/entries/edit` — guarded correction of one logged exchange (see Editing below)
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
  served alongside the corrected text;
- accepts an optional `base_hash` (sha256 hex of the exchange block from the
  `**Me:**` line up to the xid marker, trimmed; see `corpus.exchange_hash`).
  When it no longer matches, the edit is refused with `409` and
  `current_hash`/`current_text`, so two tabs cannot silently overwrite each
  other. Success responses include the new `hash` for chained edits. Omitting
  `base_hash` keeps the older last-write-wins behaviour for existing clients;
- rejects a `month` hint that is not `YYYY-MM` with `400`.

Editing is a human-initiated, visible action in noevia's web UI. The assistant never
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

`DIARY_TENANT_KEY` narrows this (M2, `docs/spec-managed-diary.md`): with it set,
every request naming a tenant must carry an `X-Cowork-Tenant-Assertion` HMAC from
the web server, bound to the tenant, method, path and storage headers, within a
60 s window and single-use. The token alone then no longer lets a caller act as
another tenant or delete one. With neither `DIARY_AUTH_TOKEN` nor
`DIARY_TENANT_KEY` set the service refuses to start unless `DIARY_ALLOW_OPEN=1`.
Keep the service internal regardless.

`TENANT_ASSERTION_IMPL=rust` (default `python`; any other value is `python` with
one warning) also runs the check in the bounded Rust leaf `tenant-assertion`
(sbstndalton/noevia-rs, baked into the image at `/usr/local/bin`). It only adds
refusals: a request is accepted when Python and Rust both accept, and a missing
binary, crash, timeout or odd output refuses it (401). The key and headers reach
the child on stdin only, with a `PATH`-only environment. `TENANT_ASSERTION_BIN`
overrides the binary path (tests).
