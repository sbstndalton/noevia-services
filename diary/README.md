# Diary Companion service

Diary Companion is Cowork's optional, independently runnable FastAPI service. It provides a browser UI and an OpenAI-compatible `/v1` API, assembles diary context for conversations, and records selected exchanges into daily Markdown files.

## Storage

`CORPUS_BACKEND=local` is the default and uses `CORPUS_LOCAL_ROOT`. `CORPUS_BACKEND=webdav` uses `WEBDAV_BASE_URL`, `WEBDAV_USERNAME`, and `WEBDAV_PASSWORD`. `CORPUS_ROOT` adds an optional subdirectory within either backend.

Both backends implement the same versioned storage contract. Local writes use locking, content hashes, `fsync`, and atomic replacement; WebDAV writes use ETag preconditions and bounded conflict retries.

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
- `GET /v1/models`
- `POST /v1/chat/completions`

Set `DIARY_AUTH_TOKEN` before network exposure. The SQLite database contains the retrieval index and durable write journal and must live on persistent storage.
