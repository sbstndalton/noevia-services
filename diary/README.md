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

The chat and aux model ids (`LLM_CHAT_MODEL`, `LLM_AUX_MODEL`) must be chat-capable ids the engine
serves, and every chat request names one explicitly. Being listed is not enough: a section the engine
launched with an embedding or reranking flag is listed but is not a chat model, so a configured id that
names one is refused like an unserved id (#112). The companion checks this at call time against the
engine's `/v1/models` (remembered for 60 s). A served configured id is used. Otherwise it uses the id
named in the optional `LLM_CHAT_FALLBACK_MODEL` / `LLM_AUX_FALLBACK_MODEL` if that is served, and logs
this once at WARNING naming the setting key, never the value. The aux role has one more substitute, the
resolved id of the chat model. It is used only after the aux id and `LLM_AUX_FALLBACK_MODEL` are both
unserved, and only when `LLM_AUX_BASE_URL` is the same base URL as `LLM_BASE_URL` (the same engine). It
follows whatever chat resolved to, including the chat fallback. It is not an explicit fallback and has no
setting of its own (#112).
It never picks "the loaded model" or "the first listed model": that would make a one-model router load
and swap out a model nobody chose (#2099). With no served id the request is refused before anything
is sent, with an error that names the setting key. Sections launched with an embedding or reranking
flag are never chat models. An engine that cannot be reached is a logged `EngineUnreachable` error.
The service also warns at startup, and `GET /api/health` lists in `model_warnings`, the setting keys
(never values) whose id is not served or is replaced by a substitute. A health check resolves each model
client once and lists each engine at most once (#112). `GET /api/health` reports `model` (the id in use),
`model_configured` and `model_source` (`configured`, `fallback`, `unverified`, `unreachable`,
`not_served` or `none_served`). The companion has no access to noevia's per-user Auto roles (they
live in the web server's workspace), so the fallback does not consult them.

Every companion answer is capped at `LLM_MAX_ANSWER_TOKENS` tokens (`llm.max_answer_tokens`, default
1024; anything that is not a positive whole number falls back to the default). An answer the engine cuts
off at the cap is still returned and saved, ending with "(This answer was shortened: it reached the answer
length limit.)"; the note is not written to the diary. The `/v1` reply then has `finish_reason: "length"`
and `diary.truncated: true`. When the 300 s exchange deadline gives up on an answer, the diary closes the
connection to the engine so it stops generating, and `/v1` answers 502 with `error.code: "chat_timeout"`
(the stream error event carries `reason: "chat_timeout"`) (#127).

The cap bounds the visible answer, not private reasoning (#164). A model that streams reasoning before its
answer gets a reasoning allowance on top of the cap: the model call is sent `max_tokens` = `LLM_MAX_ANSWER_TOKENS`
+ `LLM_MAX_REASONING_TOKENS` (`llm.max_reasoning_tokens`, default 4096; a whole number of 0 or more, anything
else falls back to the default, and 0 sends the answer cap alone). That combined `max_tokens` is the hard
generation bound. The answer cap itself is approximate: it is enforced on the visible answer as characters,
`LLM_MAX_ANSWER_TOKENS` x 4 (inline reasoning and the trailing log marker are not counted), the same way for
`/v1`, the stream and an unstreamed reply, whatever the engine chunking. A longer answer is cut exactly at that
character boundary, the stream is closed so the engine stops, and the answer counts as cut at the cap, as above
(#166). The exchange deadline still bounds the time spent.

If the cap is reached before any answer text (the model spent the whole budget, answer cap plus reasoning
allowance, on reasoning), nothing is saved
and the reason is told apart from other model failures (#135): `/v1` answers 502 with
`error.code: "length_cap"` and a message naming `LLM_MAX_ANSWER_TOKENS`, and the stream error event keeps
`code: "modelFailed"` and adds `reason: "length_cap"`. A cut answer that lost its trailing log marker is logged
by default (never lose content), saved without the shortened-answer note, and one info log line records that
the marker was missing because of truncation (no answer text).

An answer that stopped normally but holds no answer text (empty, or only the log marker) is also never saved,
and says so (#146): `/v1` answers 502 with `error.code: "empty_answer"`, the unstreamed local exchange answers
503 with `code: "modelFailed"`, `reason: "empty_answer"` and `saved: false`, and the stream error event carries
the same `code`, `reason` and `saved: false`. No `diary` event follows and the exchange is not retried.

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
  `vector_rank`, `id`, `day`, `time`, `score`, `reason` = `near_name`, `near_name_unmatched`, `other_person`,
  `below_margin`, `after_time_anchor` or
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
- A hit that could refer to an asked entity is never dropped for a name: one that names it in full (for a
  name of two or more words the surname may carry a trailing "s" or a possessive, "Alder Brannocks boat" or
  "Alder Brannock’s", which is the asked person and not a near spelling, services#148), with a
  middle name or initial or in reverse order ("Alder J. Brannock"), by first name or surname alone, by a short
  form ("Al", or one from a small table of unambiguous pairs such as "Will" for "William" or "Jan" for
  "Janet"; "Mark" is its own name, not a short form of "Marcus"), by a shared name of an entity
  that no hit names in full, or in lower case (one-word names of three or more letters too: "saw marit"). A
  near spelling is kept (and labelled) while no hit names that entity in full; once one does, it is dropped
  as `near_name`. The one exception is `near_name_unmatched` below.
- `near_name_unmatched` (services#145): when the asked entity is a two-word person that no hit names in full
  (its entry was deleted, or never written) and the question is not relational ("X's sister"), a hit naming
  the same given name with a near-spelled surname ("Caspian Juxton" for "Caspian Juxon") is dropped. The
  note says the diary does not record that person, names the near spellings (up to three), and tells the
  answer to ask whether one of them was meant instead of giving its details. Accepted trade-off: a misspelt
  real name in the question gets this abstention and a "did you mean" instead of an answer. A one-word name,
  a near given name, or a hit that also names the asked person by first name or surname alone is unaffected.
  The note stays in the context even when no other entry is loaded. A short form that is also an ordinary word ("Will", "May", "Mark", "Bill", "Jan") counts
  only where its position and the next word read as a name: "Will you ...", "Mark the date", "Bill me" and
  "in Jan 2030" name nobody, and in a question it never counts when it opens the sentence. Names compare without case, accents or letters
  such as ø, ł and ß ("Søren" = "Soren", "Strauß" = "Strauss").
- Only when some hit names an asked entity in full (a full exact hit), a hit that cannot refer to any asked
  entity is dropped as:
  - `near_name`: a near spelling of that entity; the diary names the person, so this is someone else;
  - `other_person`: the asked entity is a person (a full name, or a title and a name) and the hit's own body
    confidently names another person: a title and a name, or a full name that does not look like a place,
    organisation, product or work (a whole listed place phrase such as "San Francisco" or "Hong Kong", or an
    institutional ending such as "... Maps", "... Airport", "... University"). A given name and
    surname is always a person, whatever word precedes it ("visited Anna Pirelli"). That includes a surname-like
    ending ("Elena Bank", "Tom Hill", "Sam Green"), which marks a place or institution only after three or more
    words, a first word that is no known given name ("Northgate Bank", "Hyde Park"), a listed whole place phrase,
    or "along" or "at" right before it ("along Rosa Bank", "paid at Clara Bank"; the cue decides nothing else, and
    "waved to Elena Bank" is still a person). Mostly-place endings ("Mary Street", "Ivy House", "Grace Church") always
    mark a place. Accepted trade-off: "Rose Hill" reads as a person. A full name opening the entry or a
    sentence (list markers, quotes, dashes and words such as "Today" or "Then" are skipped) counts only with
    person evidence the module already has (services#160): a known given name first ("Tom Hardy fixed the
    boat."), or the asked person's given name, or for a two-word name the asked surname plain or with a
    trailing "s" ("Wren Brannocks fixed the boat."). Capitalised openers ("Visited Leeds", "Harbour Lights",
    "New Year", "Bank Holiday") and bold speaker labels ("**Bryony Castellow:** went") have none and are kept.
    Known limit: a capitalised verb before another person ("Met Tom Hardy.") is still read as an opener. Single words ("Marit", "Mum", "Leeds"), relation words
    and headers never count;
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
- An entry naming an asked person in full is labelled "names the asked person ... exactly" (services#178); a
  near spelling never gets that label. A name-less follow-up that refers back to the previous exchange ("and
  her email?", "Is that written in my diary?") carries the person of the previous question (services#177).
- An earlier entry is labelled superseded only by an explicit correction of the same entity that replaces a
  value found in it ("..., not smoked lapsang").

Accepted recall limits, each pinned by a test: an entry about the asked person that never names them (a
pronoun only, a nickname unrelated to the name and not in the short-form table, such as "Ziggy") is dropped when it confidently names
another person or falls below the margin; a work title read as a full name ("Moby Dick") counts as a person. When anything was dropped for not naming the asked entities, the
note says the list may be incomplete.

A short note opens the retrieved block. Messages without a name or anchor are unaffected, and
`retrieval.entity_focus: false` turns the check off.

`DIARY_TEMPORAL_FOCUS_IMPL=rust` opts the date-anchor and candidate cutoff within that check into the pinned `diary-temporal-cli` binary at `/usr/local/bin/diary-temporal-cli`. The default remains Python. The service passes the one validated native anchor to both pool selection and post-search cutoff; content/name matching remains in Python. Native capacity, protocol, process, or missing-binary failures make the exchange context unavailable before model delivery or saving, with no Python fallback. The Rust binary must be supplied and pinned separately before the selector is enabled. This adapter does not change images, pins, or runtime selection.



## Score gap (services#114)

A fixed `top_k` loads eight entries even when one or two are relevant. When the message names no entity (and is
not a follow-up carrying one), `agent/score_gap.py` drops candidates that score more than `retrieval.score_gap`
(off by default; 0.08 is the suggested value; a gap of exactly the setting is kept) below the best remaining candidate. It runs after the
entity focus, so with a time anchor the gap is measured from the best entry dated before the anchor. When the
message names an entity the entity focus alone decides, so an entry about an asked or mistyped name is never
dropped for its score. The best candidate is always kept and order never changes. The setting is a number in
(0, 1], or `off` (also `none`, `false`, 0); anything else logs a warning and leaves it off. It is off because, on the traced queries of gate run 1, it added
no precision over the entity focus, and the query families without traces are untested; enabling it waits for a
full gate run with traces for every family. The retrieval trace lists
the removed candidates under `score_gap.removed`.

`gate/replay_precision.py` replays the recorded candidates of synthetic gate run 1
(`tests/fixtures/precision_replay_114.json`) through the context assembler and prints precision@8 and recall@8
per setting and query family. It computes no embeddings, so it only shows what a setting keeps or removes from
recorded candidates; it is not a substitute for a full gate run.

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

### Workspace archive decoder source pairing

The standalone Diary image pins noevia-rs `4d18712c75be72f209af47ea420bfeb25da3dca3`
(noevia-rs#301) with verified codeload SHA-256
`0eedcd57b1852e4ed91c4db5d7cb5360a7656cc1849c3095b0760953d1015831`.
This supplies the `decode-v1` command required by the explicit Rust restore/import
path. Its versioned frame, successful exit and exact EOF are mandatory; an older
verify-only binary fails closed without Python ZIP fallback.

Both image construction and CI run `scripts/smoke_workspace_decode.py` against the
installed binary using synthetic source bytes, settings, an empty folder and an
invalid archive. The smoke disables the Python parser and selects Rust only in its
own process. The service selector default remains Python.

The existing CI gate requires Diary and OCR source pins/checksums to match. This
Diary pairing depends on the coordinated source-pin update for the other images
and release lock; the gate is expected to reject a standalone mismatched pairing
until that dependency is integrated. Do not bypass that check.

### Opt-in native managed ZIP import transaction

The default import APPLY implementation remains Python. Explicit
`WORKSPACE_IMPORT_APPLY_IMPL=rust` selects the fixed installed
`/usr/local/bin/diary-workspace-import-apply apply-v1` transaction CLI from
noevia-rs#311; this source-only adapter does not build/install that binary, change
an image pin or activate the selector. Missing/incompatible native binaries fail
closed. There is no environment-selected executable or Python mutation fallback.

Python preview and review validation finish their SQLite contexts before spawning
Rust. `ManagedCorpusBackend.close()` is a no-op and is not used for this purpose.
The application keeps its existing store write/migration/journal exclusions until
the child exits and is reaped. Rust receives the original ZIP, reviewed fingerprint,
normalized destination and the backend-derived database path/tenant/debounce;
verified source application belongs to its single SQLite transaction.

The versioned IPC caps metadata at 1 MiB, original ZIP at the existing 272 MiB
limit, and stdout/stderr at 4096 bytes each. Browser uploads retain their 32 MiB
limit. Nonblocking pipes concurrently feed input and drain output under a
120-second monotonic deadline; faults kill/reap only the owned child. Responses
require exact fields, version, destination/count, EOF and successful exit.
Child output and private paths/bytes are never logged.

Preview validation remains HTTP 400; APPLY validation/conflicts remain 409.
Native unavailability and uncertain outcomes return 503 without draining the
index outbox or claiming success. An uncertain outcome clears the recovery flag
because a receipt/outbox may already have committed. Retry the same archive,
destination and reviewed fingerprint: the durable receipt makes that retry
idempotent. Normal later recovery transfers/indexes pending outbox entries; the
import itself never invokes a model.

Synthetic tests in `tests/test_workspace_import_native.py` use the reviewed Rust
binary when `WORKSPACE_IMPORT_APPLY_TEST_BIN` is supplied to pytest. That variable
is test-only and never changes production executable selection. The tests cover
transaction closure, locks through reap, tenant/deleted/replaced databases,
rollback, lost replies/idempotent retry, bounded child failures, HTTP status and
exactly-once later outbox transfer. No real Diary data is required.
