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

| | |
|---|---|
| Models | layout ~1 GB + TableFormer ~600 MB, baked into the image |
| Speed | ~3 s/page on x86 CPU per Docling's technical report — **their** benchmark box, not yours; measure with `selftest.py` |
| GPU | none. No device is mapped, so it cannot contend with llama.cpp |

CPU-only is deliberate, not a limitation: it is what lets extraction run
concurrently with inference on a single-GPU host.

Two things are deliberately **not** installed:

- **EasyOCR**, Docling's default OCR engine, measures ~13 s/page on CPU and
  would dominate everything else. Tesseract is used instead — already in the
  image, and what the previous OCR worker used, so output stays comparable.
- **The chart pipeline.** Chart extraction loads
  `granite-vision-3.3-2b-chart2csv`, a 2B vision model — a different order of
  cost from the ~1.6 GB layout/table pair, and it would contend for VRAM. It is
  a separate decision, and enabling it means revisiting `_build_converter` in
  `extract.py`, the memory limit in the compose overlay, and where the job
  queues.

## Verify it on your own hardware first

`extract.py` was written against Docling's documented API but **was not
executed against a real Docling install** during development — the build
environment could not reach HuggingFace for the models. Everything around the
conversion is unit-tested with Docling stubbed; the conversion itself is not.

    docker compose -f compose.yaml -f compose.docling.yaml run --rm \
      --entrypoint python docling selftest.py /path/to/sample.pdf

It prints per-page timings, peak RSS, page statuses, and the first lines of
each page, so reading order and table structure can be checked by eye. Run it
against a multi-column PDF and a spreadsheet before trusting the sidecar — per
`docs/agent-brief.md`, a green test suite is not sufficient evidence here.

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
