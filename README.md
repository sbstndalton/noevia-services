# noevia-services

The noevia Python sidecars, one image build context each: `diary/`, `docling/`, `laya/`,
`model-manager/`, `ocr/`. Their image tags (`DIARY_VERSION`, `OCR_VERSION`,
`MODEL_MANAGER_VERSION`, `DOCLING_VERSION`, ...) keep their meaning; the code sandbox lives in
[noevia-core](https://github.com/sbstndalton/noevia-core).

Split out of [sbstndalton/noevia](https://github.com/sbstndalton/noevia) `services/` by
`tools/repo-split` (issue #952, ADR 0001). Every file except this README and `.github/` is noevia
history filtered to those paths; the last commit names the noevia SHA it was cut from
(`Split-Source:`).

This repo is the source of truth for these paths since the cutover (noevia #952, cut at noevia
`f42f65f1`). A release uses the SHA pinned as `NOEVIA_SERVICES_REF` in noevia's
`release/versions.lock`; bump it there to ship a change made here.

Tests use synthetic fixtures only. Never send prompts to the real Diary or touch its corpus.

## CI

`.github/workflows/ci.yml` builds a noevia-shaped workspace (noevia `main` for the integration
files such as `docs/spec-model-loader-api-v1.md` and `release/versions.lock`, noevia-web and
noevia-core `main`, this checkout on top) and runs each sidecar's test suite, the gguf-meta
differential test, and the diary and model-manager image builds.
