# docling worker

Document extraction for noevia's project sources. **Opt-in**: with
`DOCLING_BASE_URL` unset, `apps/web/server/documents.cjs` keeps its pdf.js path
and nothing here runs.

    docker compose -f compose.yaml -f compose.docling.yaml up -d

## Why

The previous extractor walked pdf.js text items in native order. That meant:
one format (`.pdf`), multi-column pages interleaved, no table structure, and
`.xlsx`/`.pptx`/`.odt`/`.epub` accepted into the Documents folder and then
silently stored with empty content.

Docling brings reading order, TableFormer table structure, and those formats.
Tables come back as Markdown, so the structure survives into the chunker and
into the model's context — which is the point.

## What it costs

Measured on DaServer (Unraid, 2 CPUs allocated, CPU-only torch), 2026-09-20 —
these are observations from this hardware, not vendor figures:

| | |
|---|---|
| Models | 669 MB total (`layout` + `tableformer`), baked into the image |
| Speed, book prose (native text) | **1.9 s/page** |
| Speed, scanned prose (OCR) | **3.4 s/page** |
| Speed, table-heavy native PDF | **10.9 s/page** |
| Speed, spreadsheet | **0.9 s/sheet** |
| Peak RSS, small documents | 1.8–2.0 GB |
| Peak RSS, 334-page PDF | **4.79 GB** (steady across pages, not a leak) |
| GPU | none. No device is mapped, so it cannot contend with llama.cpp |

Two results worth knowing before you tune anything:

- **TableFormer, not OCR, is the cost driver.** Per-page cost spans 1.9 s
  (plain book prose) to 10.9 s (dense tables) — a 5.7x spread driven by table
  structure, not by OCR. Scanned pages sit in the middle at 3.4 s. The
  intuition that OCR is the expensive path is wrong here, so budget by how
  table-heavy the corpus is, not by how much of it is scanned.
- **The earlier "~3 s/page" figure was Docling's own benchmark box.** It
  happens to bracket the middle of the real range, but table-heavy pages miss
  it by 3.5x, which is the case that matters for a memory limit and a timeout.

OCR quality on a printed scan was good: 7,771 characters recovered against
7,906 in the source text layer (~98%), reading order intact.

CPU-only is deliberate, not a limitation: it is what lets extraction run
concurrently with inference on a single-GPU host.

Two things are deliberately **not** installed:

- **EasyOCR**, Docling's default OCR engine, measures ~13 s/page on CPU and
  would dominate everything else. Tesseract is used instead — already in the
  image, and what the previous OCR worker used, so output stays comparable.
- **The chart pipeline.** Chart extraction loads
  `granite-vision-3.3-2b-chart2csv`, a 2B vision model — a different order of
  cost from the 669 MB layout/table pair, and it would contend for VRAM. It is
  a separate decision, and enabling it means revisiting `_build_converter` in
  `extract.py`, the memory limit in the compose overlay, and where the job
  queues.

## Verified on DaServer, 2026-09-20

`extract.py` was written against Docling's documented API and not executed
during development — the build environment could not reach HuggingFace for the
models. It has since been **built and run on DaServer** against a table-heavy
native PDF, a scanned PDF, a scanned-printed PDF with known ground truth, a
generated .xlsx, and a 334-page book, under the exact production constraints
(`read_only`, non-root 65534, `cap_drop: ALL`, tmpfs, memory limit).

Three defects were found and fixed in the process; none of them were in the two
functions the original note predicted (`_convert` / `_pages_from` were
substantially right):

1. **The image did not build.** `docling-tools models download` with no
   arguments resolves its whole default set, including the PP-OCR/rapidocr
   recognizers, and rapidocr is not installed — `ImportError`, build dead.
   Fixed by naming `layout tableformer`, which is also all this pipeline uses.
2. **TableFormer could not start.** `cv2` is imported by
   `docling_ibm_models`' `tf_predictor` and no chosen extra pulls in OpenCV, so
   the converter raised `ModuleNotFoundError` the moment table structure was
   enabled — i.e. the feature this service exists for. Fixed with
   `opencv-python-headless`.
3. **Large documents were OOM-killed.** `PAGE_CAP` truncated *output* while
   Docling still converted every page, so a 334-page PDF did full work and was
   SIGKILLed at the 4 GB limit after 520 s. `_convert` now passes
   `page_range=(1, PAGE_CAP)` so the cap bounds the work, and the limit is 6g.

The client timeout in `apps/web/server/docling.cjs` was also raised: at 610 s it
covered under a fifth of the 300-page worst case it claimed to survive.

    docker compose -f compose.yaml -f compose.docling.yaml run --rm \
      --entrypoint python docling selftest.py /path/to/sample.pdf

It prints per-page timings, peak RSS, page statuses, and the first lines of
each page, so reading order and table structure can be checked by eye. Run it
against a multi-column PDF and a spreadsheet before trusting the sidecar — per
`docs/agent-brief.md`, a green test suite is not sufficient evidence here. All
three build/runtime defects above passed the unit suite untouched.

## Known gap: `blank` does not mean empty

A page that yields no text is reported `status: "blank"`. That is **not** the
same as the page having no content, and the code deliberately does not guess.
Measured on a page of handwritten notes: Tesseract cannot read handwriting,
Docling's layout model returned zero clusters, and the page arrived
indistinguishable from a genuinely blank one — 2 pages of dense handwriting
produced 32 characters. Nothing in the conversion result separates "empty" from
"unreadable", so callers must read `blank` as *no text recovered*. Handwriting
is out of scope for Tesseract; a document of it will extract as near-nothing
with no error raised anywhere.

## Native-text fallback (#700)

The opposite failure to the gap above: a PDF page with a real text layer that
Docling returns empty. Found in the #262 verification — 1 page of 66 had about
5k characters of native text and no Docling output, was reported `blank`, and
the document showed as ready. One way it happens is a page Docling's layout
files as a single picture region, whose text it then skips.

For PDFs, any page where Docling kept fewer than
`DOCLING_NATIVE_FALLBACK_MIN_CHARS` (default **200**) non-whitespace
characters has its own text layer read with pypdfium2 (already in the image
through the `format-pdf` extra). If that layer holds at least the threshold
and Docling kept under 10% of it, the native text is used and the page is
reported `degraded` with `reason: native-fallback`: the words are right, but
reading order and tables are pdfium's raw order, not Docling's layout. The web
app shows such a document as partially extracted.

Deliberately not done: walking the text Docling files under picture regions
(`traverse_pictures=True`). That would change every page with a figure — chart
labels, axis ticks, logo text — in every document, and fix only one of the ways
a page can come back empty. The fallback touches only pages that would
otherwise lose their text, whatever the cause. A page with no text layer
(a scan, handwriting) still has nothing to fall back to and stays `blank`.

Tests build synthetic PDFs at run time (`synthetic_pdfs.py`): text over a
full-page image, text only, a blank page, and an image with no text.

Shipping it takes a docling image release (`DOCLING_VERSION`), and it goes
**before** the web release that carries it: web's `docling.cjs` VERSION moved
to `…-v3-native-fallback`, so every Docling-read document is re-extracted on
its next sync, and a re-extraction against the old image would cache the old
`blank` page under the new key. To check the fallback on the host without a
real document:

    docker compose -f compose.yaml -f compose.docling.yaml run --rm --entrypoint sh \
      docling -c 'cd /tmp && python /app/synthetic_pdfs.py && python /app/selftest.py synthetic-picture-text.pdf'

A page reported `degraded` or `native` with the widget-audit text is a pass;
`blank` is the #700 bug.

## Posture

Same as `services/ocr`, and for the same reasons: stateless, no volume, no
corpus credentials, one job at a time, non-root, read-only root filesystem,
and **no logging of document contents or names**. Errors returned to the caller
are generic because exception text can quote document content.

`HF_HUB_OFFLINE=1` at runtime: the models are fetched at image build time, so a
conversion can never stall on a download and the container works on a host with
no outbound access.

## Reproducible build and release path (#890)

The image build is pinned end to end: `requirements.txt` and `requirements-torch-cpu.txt` are
hash-locked (`pip install --no-deps --require-hashes`; edit `requirements.in` and regenerate, the
header of `requirements.txt` has the command), and `download_models.py` fetches the layout and
TableFormer models at exact Hugging Face commits instead of the floating `main` that
`docling-tools models download` resolves. `test_requirements_pinned.py` fails on any unpinned
package. Releases that only change the six application files do **not** rebuild: they ship as an
overlay FROM the running image, documented in `docs/deployment.md` ("Docling release path").

## Contract

`POST /extract`, body = raw bytes, `X-Document-Name` header. Returns:

    {"pages": [{"number": 1, "text": "...", "status": "native",
                "method": "docling", "truncated": false}],
     "total": 12, "truncatedPages": false}

### Figures (#1726, multimodal RAG)

With `X-Document-Figures: 1` (sent by noevia-core only while `NOEVIA_MULTIMODAL_RAG` is on) the reply
also carries the document's figures, and every page says whether its text came from OCR:

    {"pages": [{..., "ocr": false}, {..., "ocr": true}],
     "figures": [{"page": 1, "index": 0, "mime": "image/png", "data": "<base64>",
                  "caption": "Figure 1: ...", "text": "words inside the picture"}], ...}

- `ocr: true` marks a scanned page: a PDF page whose own text layer has fewer than 20 visible
  characters although Docling returned text for it (so that text is OCR), or any page of an image
  input with text. Core embeds such a page as text only, so its figures are **never returned**.
- Figures are each picture Docling found (`generate_picture_images`, rendered at scale 2.0 for PDFs),
  as PNG at most 1024 px on the longer side; pictures under 64 px on a side are skipped. At most 64
  per document and 12 MiB of base64 in all, so the reply stays inside the 32 MiB frame. An image
  input returns none (it is its own image).
- A second converter carries the picture rendering, built on first use, so a request without the
  header converts exactly as before and its reply is unchanged.

### Per-document limits (#854)

Every input type has a wall-clock limit, enforced by **killing the conversion**, not by asking
Docling nicely: each conversion runs in a long-lived child process (`isolated_worker.py`, started
by the native server) that holds the converter and its models, so they load once. The server kills
the child's whole process group (taking any `tesseract` it started with it) when

- the deadline passes: **3600 s** for `.pdf`/`.tif`/`.tiff` (`DOCLING_DOCUMENT_TIMEOUT_SECONDS`,
  kept under the web client's 65 min abort so it receives an answer), **600 s** for everything
  else (`DOCLING_OTHER_TIMEOUT_SECONDS`) -> `422`;
- the requester disconnects -> nothing is sent, the slot is freed at once;
- the child dies on its own (an OOM kill) -> `422`; the next document starts a fresh child, which
  pays one model load.

Docling's own `document_timeout` is deliberately not used: only its PDF pipelines honour it, and
when it fires it returns a partial result, which would show the missing pages as `blank` on a
document reported ready. Non-PDF inputs also get `max_num_pages=1000` (sheets, slides, frames; a
document over it is rejected before any work) and every input `max_file_size=25 MB`. A PDF is still
converted up to `PAGE_CAP` pages of any length.

### Page ranges (#1864, #1867)

With `X-Document-Pages: first-last` (PDFs only, at most 200 pages, `first >= 1`) only that range is
converted, and the reply holds only those pages; `total` is still the whole document's page count and
`truncatedPages` says whether pages follow the range. Such a request may carry a file up to
`DOCLING_RANGED_MAX_BYTES` (default 96 MiB, noevia-core's `PROJECT_DOCUMENT_READ_BYTES` default) instead
of 25 MB, streamed to `/tmp` in 1 MiB chunks (30 s per read, and 30 s plus one second per 256 KiB in all),
never held in memory. `/health` answers `"ranges": true` and `"rangedMaxBytes"`; noevia-core asks before
it sends a range, so a worker without this is never sent one. noevia-core reads a PDF over 25 MB, and the
pages of any PDF past the first 300, this way. The native server (noevia-rs `docling-service`)
implements this contract (#1867).

`status` ∈ `native | blank | truncated | degraded`. A `degraded` page also
carries `"reason": "native-fallback"` and `"method": "pdfium"` (see below). 415 = format this worker cannot read,
422 = document could not be read or exceeded its time limit, 503 = busy. The client
(`apps/web/server/docling.cjs`) treats 415/422 as permanent — cache the
failure, keep the original — and everything else as retryable.

## Native HTTP and worker supervision (#1322)

The checksum-pinned `native-docling` Docker stage produces
`/out/noevia-docling` with `native-docling` compiled, plus
`/out/noevia-docling-omission` without that capability. The production image
uses `noevia-docling` as its entrypoint: the native HTTP server and process-group
worker supervisor, the only owner since noevia#1071. The Python HTTP server
(`server.py`) and its supervisor (`isolation.py`) are retired and deleted; they
survive only as the test oracle, the merged copy at services `bbed9b8` that the
noevia-rs differential and recorded goldens run against. `NOEVIA_DOCLING_IMPL`
is retired: the images fix it to `rust`, which the pinned binary still reads,
and a binary built after the retirement ignores it, warning once about any
other value. The Python extraction engine (`extract.py`, `isolated_worker.py`)
remains the same. Missing native capability refuses before binding or spawning
a worker.

The existing engine/dependency/model-download layers are unchanged. Ordinary
release packaging must use the documented app overlay onto the existing engine
image, with the Bookworm-compatible compiled binary; rebuilding model layers is
not part of native interface qualification.

The unconditional Docling CI job builds only `native-contract` and
`omission-contract`, using the native server, the actual worker and a fixed
synthetic converter. It checks the native owner by default and when set, warm-worker reuse,
generic conversion errors, full/half-close cancellation, temporary-file cleanup,
nonroot/read-only/network-none/resource isolation, and specific feature omission.
The Rust host guards run from `self/docling` so they inspect the actual PR
workflow, outside the assembled workspace and Docker image. No real model is
imported, downloaded, loaded, or exercised by these targets. These checks qualify
the interface and supervisor, not production ML extraction accuracy.

Native admission is deliberately bounded: 25 MiB input, 200-byte document names,
32 MiB framed output, one conversion and eight admitted connections. Existing
Rust worker/oracle tests cover output/depth/deadline limits and the stricter
refusal boundaries.
