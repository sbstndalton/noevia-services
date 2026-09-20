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

## Posture

Same as `services/ocr`, and for the same reasons: stateless, no volume, no
corpus credentials, one job at a time, non-root, read-only root filesystem,
and **no logging of document contents or names**. Errors returned to the caller
are generic because exception text can quote document content.

`HF_HUB_OFFLINE=1` at runtime: the models are fetched at image build time, so a
conversion can never stall on a download and the container works on a host with
no outbound access.

## Contract

`POST /extract`, body = raw bytes, `X-Document-Name` header. Returns:

    {"pages": [{"number": 1, "text": "...", "status": "native",
                "method": "docling", "truncated": false}],
     "total": 12, "truncatedPages": false}

`status` ∈ `native | blank | truncated`. 415 = format this worker cannot read,
422 = document could not be read, 503 = busy. The client
(`apps/web/server/docling.cjs`) treats 415/422 as permanent — cache the
failure, keep the original — and everything else as retryable.
