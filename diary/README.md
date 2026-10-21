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

The chat and aux model ids (`LLM_CHAT_MODEL`, `LLM_AUX_MODEL`) must be ids the engine serves, and
every chat request names one explicitly. The companion checks this at call time against the engine's
`/v1/models` (remembered for 60 s). A served configured id is used. Otherwise it uses the id named in
the optional `LLM_CHAT_FALLBACK_MODEL` / `LLM_AUX_FALLBACK_MODEL` if that is served (the aux role then
follows the chat model's id), and logs this once at WARNING naming the setting key, never the value.
It never picks "the loaded model" or "the first listed model": that would make a one-model router load
and swap out a model nobody chose (#2099). With no served id the request is refused before anything
is sent, with an error that names the setting key. Sections launched with an embedding or reranking
flag are never chat models. An engine that cannot be reached is a logged `EngineUnreachable` error.
The service also warns at startup, and `GET /api/health` lists in `model_warnings`, the setting keys
(never values) whose id is not served. `GET /api/health` reports `model` (the id in use),
`model_configured` and `model_source` (`configured`, `fallback`, `unverified`, `unreachable`,
`not_served` or `none_served`). The companion has no access to noevia's per-user Auto roles (they
live in the web server's workspace), so the fallback does not consult them.

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

## Retrieval trace (test harness only)

For the synthetic Diary gate. Off unless the service environment sets `DIARY_RETRIEVAL_TRACE=1` (production never
does) **and** the request sends `X-Diary-Retrieval-Trace: 1`. Then `/v1/chat/completions` returns a
`retrieval_trace` object (top level of the JSON response; in the `diary` event when streaming with `diary_events`).
It is built per request and returned only to that caller; nothing is stored and there is no trace endpoint.
It holds ids, scores, counts and the budget only, never entry text, headers, the query or embeddings, and it does
not change retrieval or the model input.

- `candidates`: the candidates offered for selection, in score order (`rank`, `vector_rank`, chunk `id`, `day`,
  `time`, `score`, `loaded`, and `dropped` = `budget` or `top_k` when not loaded). `params`: `top_k` 8,
  `candidate_k` (vector hits requested: 16 = 2x top_k, or 32 when the entity focus runs), `min_score` 0.30.
- `focus` (services#116): `{"checked": false}` when the message names no entity and no time anchor; then
  `candidates` are the vector hits. Otherwise `entities` and `time_anchor` (counts and flags only, never the
  names), `applied` (some hit named an asked entity exactly), `vector_candidates`, `removed` (each dropped hit:
  `vector_rank`, `id`, `day`, `time`, `score`, `reason` = `near_name`, `other_person`, `below_margin`,
  `after_time_anchor` or
  `candidate_limit`), `removed_by_reason` (counts), `carried` (the entities came from the previous
  question), `relational`, `margin`, `similar_name_labelled` and `superseded_labelled`.
- `rerank`: `{"present": false}`. The diary has no reranker; selection is score order under the token budget.
- `loaded`: the excerpts actually placed in the context (`id`, `day`, `time`, `score`, `tokens`, `chars`);
  `budget`: `max_context_tokens`, `max_context_chars`, `excerpt_char_cap`, `used_chars`, `used_tokens`.
- `counts`; `status` (`ok`, `no_retriever`, `empty_query`, `search_failed`, `index_unavailable`, `embed_failed`, `embed_dim_mismatch`, `search_error`; only `ok` means the search ran); `direct_fallback` (`used` = it ran, `found` = it produced text, `tokens`). If tracing itself fails the chat still succeeds and the response carries no trace.

## Entity focus (services#116)

Vector similarity cannot tell similarly spelled names apart, so entries about "Alder Brennock" sit next to the one
about "Alder Brannock" and the answer conflates them. When the user's message names an entity (a capitalised name
phrase; titles, accents and punctuation ignored), is a name-less follow-up ("and her email?") to a question that
did, or is explicitly framed by a date ("As of January 20, 2030, ...?"), `agent/entity_focus.py` checks a wider
vector pool (4x top_k) lexically, with no model call, and never reorders it:

- `after_time_anchor`: hits dated after an explicit time anchor are dropped. A date inside a statement ("away
  until 3 March") is not an anchor.
- A hit that could refer to an asked entity is never dropped for a name: one that names it in full, with a
  middle name or initial or in reverse order ("Alder J. Brannock"), by first name or surname alone, by a short
  form ("Al"), by a near spelling or a shared name of an entity that no hit names in full, or in lower case
  (one-word names of three or more letters too: "saw marit"). Names compare without case, accents or letters
  such as ø, ł and ß ("Søren" = "Soren", "Strauß" = "Strauss").
- Only when some hit names an asked entity in full (a full exact hit), a hit that cannot refer to any asked
  entity is dropped as:
  - `near_name`: a near spelling of that entity; the diary names the person, so this is someone else;
  - `other_person`: the asked entity is a person (a full name, or a title and a name) and the hit's own body
    confidently names another person: a title and a name, or a full name that does not open a sentence (list
    markers, quotes and dashes are skipped: "- Visited Leeds", "1) Called Mum") and does not look like a place,
    organisation or work. Single words ("Marit", "Mum", "Leeds"), relation words and headers never count;
  - `below_margin`: it scores more than `retrieval.entity_focus_margin` (default 0.05, `MARGIN`; a gap of exactly
    the margin is kept) below the best full exact hit. It arms only for an asked name of two or more words or a
    titled name ("Alder Brannock", "Hollin Moor", "Dr Patel"), never for a lone word ("Christmas", "Paris",
    "Python"), and only while every asked entity has a full exact hit. A two-word place or product ("San
    Francisco") arms it too. This is a deliberate precision/recall trade-off: an unnamed entry about the person
    (a pronoun only) that scores more than the margin below the best exact hit is dropped. The setting must be
    a number in [0, 1]; anything else logs a warning and uses 0.05.
- With no full exact hit nothing is dropped for a name (a typo or a short form in the question); similar names
  are labelled "may be a different person, check the name". A relational question ("Alder Brannock's sister",
  "sister-in-law", "the wife of Alder Brannock") drops nothing for not naming the asked entity.
- An earlier entry is labelled superseded only by an explicit correction of the same entity that replaces a
  value found in it ("..., not smoked lapsang").

Accepted recall limits, each pinned by a test: an entry about the asked person that never names them (a
pronoun only, a nickname unrelated to the name such as "Bob" for "Robert") is dropped when it confidently names
another person or falls below the margin; a work title read as a full name ("Moby Dick") counts as a person. When anything was dropped for not naming the asked entities, the
note says the list may be incomplete.

A short note opens the retrieved block. Messages without a name or anchor are unaffected, and
`retrieval.entity_focus: false` turns the check off.

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

The signature, the clock window and the bounds on every field are checked by the bounded Rust leaf
`tenant-assertion` (sbstndalton/noevia-rs, baked into the image at `/usr/local/bin`) and by nothing else: since
noevia#1071 there is no Python copy of those rules and `TENANT_ASSERTION_IMPL` is retired (`rust` and empty are
silent, any other value is ignored with one warning at start). A request is accepted only when the leaf accepts;
a missing binary, crash, timeout or odd output refuses it (401). This module keeps what the stateless leaf cannot
know: the shape of the header, the restart guard (an assertion signed before this process started is refused, as
the nonce cache is in memory) and the nonce cache (each nonce is accepted once). The leaf also refuses some inputs
the Python check used to accept: a tenant id that is not ASCII or is over-long, whitespace or newlines in the
storage headers, over-long fields (the tenant ids web signs are UUIDs). A `secretRef` that does not match is a
mismatch (400), a non-ASCII one no longer an error. The key and headers reach the child on stdin only, with a
`PATH`-only environment. With `DIARY_TENANT_KEY` set and no usable binary the service logs an error at start and
refuses every tenant request. `TENANT_ASSERTION_BIN` overrides the binary path (tests).
