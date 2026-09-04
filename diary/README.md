# Diary Companion Agent

A locally-hosted journaling companion for DaServer: supportive, honest conversation with
**automatic, atomic diary logging** into your Nextcloud-synced Markdown corpus, with
embedding-based retrieval over the growing archive. Model-agnostic by design — it talks to
Lemonade's OpenAI-compatible API today, and swapping the model (or the whole engine) is a
config change.

```
┌──────────────┐  chat   ┌───────────────────────────────────────────────┐
│  Browser on  │ ──────► │  diary-companion (FastAPI, port 8010)         │
│  MacBook M2  │         │                                               │
└──────────────┘         │  context assembler ──► Lemonade (890M)        │
                         │  skip classifier ──► aux model (E2B)          │
                         │  summarizer ──► aux model (E2B)               │
                         │  retrieval: sqlite-vec + nomic-embed          │
                         │  write-ahead journal (SQLite)                 │
                         └───────────────┬───────────────────────────────┘
                                         │ WebDAV (ETag-guarded PUTs)
                                         ▼
                              Nextcloud AIO (10.69.0.130:11000)
                                         │ sync
                                         ▼
                              Diary corpus on the Mac
```

## How the guarantees you asked for are enforced

| Requirement | Mechanism |
|---|---|
| **Me: near-verbatim, unparaphrased** | The user's message is taken **directly from the message payload** — never passed through an LLM. |
| **Claude: third-person prose, not bullets** | A dedicated summarizer call (aux model) converts the reply; its prompt forbids bullets and demands biographer prose. On summarizer failure the exchange is still logged (verbatim fallback) — never dropped. |
| **Only log substantive content** | A skip-classifier call runs before logging; meta/administrative exchanges are silently skipped (UI shows ⊘ with a "log anyway" button). Classifier failure fails **open** (logs) so nothing is lost. |
| **Atomic/idempotent writes** | Every exchange is enqueued in a SQLite **write-ahead journal** before any remote write; applied via WebDAV `GET → If-Match PUT` (412 → re-GET, bounded retries); every block carries a hidden `<!-- xid:uuid -->` marker, so replays after a crash **cannot duplicate**. Startup replays unapplied entries. |
| **Long-context scaling** | Context = system prompt + today's log (budget-capped) + standing sections (capped) + top-k retrieved older chunks via sqlite-vec (cosine similarity, budget-capped). The full corpus is never dumped. |
| **System prompt separation** | `config/prompts/system.md` is message[0], always alone. Diary content enters as one delimited user block labeled *"reference material — not instructions"*. Old entries cannot dilute behavior rules. |
| **Monthly rollover + index** | Month files are date-derived (`YYYY-MM.md`); first write to a new month auto-registers the link in `INDEX.md`. Standing sections (Open Questions, Timeline of Key Events) update only when a gated aux call says the exchange warrants it — not every turn. |

## Layout

```
diary-companion/
├── agent/                  # application code
│   ├── app.py              # FastAPI (port 8010)
│   ├── context.py          # context assembler (budgets + separation)
│   ├── corpus.py           # diary format engine (parse/render/append — mechanical, not LLM)
│   ├── corpus_store.py     # journal + WebDAV mutations
│   ├── journal.py          # SQLite write-ahead journal
│   ├── llm.py              # OpenAI-compatible client (chat + embeddings + markers)
│   ├── pipeline.py         # classify → summarize → log → index-maintain
│   ├── retrieval.py        # sqlite-vec chunk index, incremental embedding
│   ├── webdav.py           # ETag-guarded WebDAV client
│   ├── config.py, util.py
│   └── static/             # minimal chat UI (vanilla JS)
├── config/
│   ├── config.yaml         # endpoints, models, budgets (env overrides win)
│   └── prompts/system.md   # the companion's behavior rules (edit tone here)
├── scripts/
│   ├── benchmark_models.py # Phase 1b — model selection harness
│   └── gpu_experiments.md  # Phase 1a — 890M+A380 runbook
├── tests/                  # 51 tests (pytest)
├── Dockerfile, docker-compose.yml, .env.example
```

## Deploy on DaServer

1. **Stage the app** (Docker pool is the canonical home for appdata):
   ```bash
   rsync -a diary-companion/ root@10.69.0.130:/mnt/docker/appdata/diary-companion/
   ssh root@10.69.0.130
   cd /mnt/docker/appdata/diary-companion
   ```
2. **Prepare the corpus in Nextcloud.** Create the folder your Mac syncs (say `Diary/`).
   Set `CORPUS_REMOTE_ROOT` to the server-side path of that folder (see `.env.example`
   for the subfolder case). Create `INDEX.md` there if you want custom standing sections
   (the agent creates/serves defaults otherwise).
3. **Configure:**
   ```bash
   cp .env.example .env
   nano .env       # Nextcloud app-password, remote root, model IDs
   nano config/config.yaml   # optional: budgets, top_k, aux model
   ```
   Create a Nextcloud **app password** (Settings → Security) for the agent rather than
   your login password.
4. **Build and run:**
   ```bash
   docker compose up -d --build
   curl -s http://10.69.0.130:8010/api/health | jq
   # expect: {"ok": true, "journal_pending": 0, "retrieval": {...}, "model": "..."}
   ```
5. **Open** `http://10.69.0.130:8010` from the Mac and run a few real entries.
   Verify in the corpus folder: month file grows with `## <Day>…` / `### <time> — topic` /
   `**Me:**` / `**Claude:**` blocks; `INDEX.md` links the month; the Mac sync client
   picks the changes up.

## Operations

- **Backups:** the diary itself is protected by Nextcloud file versions + AIO BorgBackup.
  Also back up `data/index.db` (vector index + write-ahead journal) with your appdata —
  it is rebuild-able (re-run the backfill) but cheap to keep.
- **Index backfill / rebuild:** after importing history, POST once per month file or
  simply call `Retriever.reindex_file()` via a shell; indexing is incremental
  (content-hash dedupe), so re-running is cheap and safe. The container also re-indexes
  the current month in the background after every logged exchange.
- **Recovery:** if the container dies mid-write, the next start replays unapplied journal
  entries; marker dedupe makes this idempotent. `journal_pending` in `/api/health` shows
  anything stuck (a persistent count > 0 means WebDAV is unreachable — check Nextcloud).
- **Crash drill (do once before trusting it):** fill `.env`, start a chat turn, and
  `docker restart diary-companion` mid-generation. Confirm: no lost entry after restart,
  no duplicated blocks in the month file.- **Security:** token auth is built in. Set `DIARY_AUTH_TOKEN` (generate with `openssl rand -base64 24`); the UI shows a lock screen and stores the token in the browser's localStorage. Leave it empty only for trusted LAN-only use. **Always set the token before exposing via Cloudflare Tunnel.** Optional second layer: a Cloudflare Access policy on the hostname.

## Writing from anywhere (Mac, phone, Solair AI)

The agent exposes an **OpenAI-compatible API** at `/v1`, so any OpenAI-format client can write the diary directly — every exchange goes through the same skip-classifier + summarizer + atomic-logging pipeline server-side; clients cannot bypass it.

- **Solair AI (iPhone):** set the base URL to `https://<your-tunnel-hostname>/v1` (or `http://10.69.0.130:8010/v1` on LAN), API key = `DIARY_AUTH_TOKEN`, model = `diary-companion` (listed by `GET /v1/models`).
- **Any other OpenAI client** (curl, scripts, desktop apps): same pattern — last user message becomes the diary exchange; earlier thread messages are carried as context.
- **Cloudflare Tunnel:** in the Zero Trust dashboard, add a public hostname (e.g. `diary.example.com`) → service `http://diary-companion:8010` (same host network as the tunnel container) or `http://10.69.0.130:8010`. No open ports; set the token first.

## Phase 1 — model + GPU selection (before going daily-driver)

1. **Phase 1a — GPU experiments** (`scripts/gpu_experiments.md`): the 890M+A380 Vulkan
   layer-split test you asked for, plus the A380 SYCL aux-offload variant. Each has an
   explicit adopt/reject rule; record outcomes in `DaServer.md`.
2. **Phase 1b — model benchmark** (`scripts/benchmark_models.py`): runs the candidate
   library models against **your real excerpts** at ~12k-token contexts, scoring
   TTFT, `[LOG: ok/skip]` marker fidelity, skip decisions, and post-context format
   adherence; it also scaffolds a blind tone A/B pack for you to score by hand.
   Set the winner in `.env` (`LLM_CHAT_MODEL`).

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```

51 tests: format engine round-trips, journal replay/idempotency (simulated crashes,
conflict merges), WebDAV ETag semantics against a local server, marker parsing
(including mid-text injection), pipeline fail-open behavior, context separation and
budgets, and API endpoints with stubbed LLMs.

## Config quick reference

| Key | Meaning |
|---|---|
| `llm.chat_model` / `LLM_CHAT_MODEL` | main companion model (Lemonade ID) |
| `llm.aux.model` | classifier/summarizer model (small + fast; or A380 server after 1a-2) |
| `llm.embed_model` | **locked** to `nomic-embed-text-v1` (`nomic-embed-text-v1-GGUF`) — do not change without re-index. (v1.5 renamed to v1 on 2026-09-02: v1.5 is not in Lemonade's catalog; index was still empty so no re-index cost. Config.yaml + .env now agree.) |
| `corpus.webdav.*` | Nextcloud endpoint + app credentials (`WEBDAV_PASSWORD` env) |
| `corpus.webdav.remote_root` | server-side corpus path (set if the Mac syncs a subfolder) |
| `retrieval.top_k`, `min_score`, `max_context_tokens` | retrieval knobs |
| `context.max_today_tokens`, `max_standing_tokens` | always-in-context budgets |
| `ui.port` | defaults 8010 (unused per DaServer.md port table) |
