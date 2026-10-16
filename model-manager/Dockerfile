# syntax=docker/dockerfile:1.7
# noevia model manager: Model Loader (scratchhax/model-loader, MIT) folded into noevia.
# JSON API only (/api/v1); noevia web renders every screen.

# gguf-meta (#909) and model-files (#964): bounded Rust leaves from sbstndalton/noevia-rs,
# pinned to a merged NOEVIA_RS_REF and fetched as an anonymous tarball whose
# sha256 is checked before anything is built. gguf-meta is used only when GGUF_PARSER=rust (default
# python). model-files is always used (noevia#1071: repo file lists and models.ini backups; the app
# refuses to start without it). model-autoconfig only when MODEL_AUTOCONFIG=rust (default python;
# it only ever confirms the Python answer).
# Native health/logs (#1319, noevia#1071): the noevia-model-manager entrypoint always owns them and
# supervises the Python app on private loopback 8091; there is no Python-only mode.
# Native section GET reads require MODEL_MANAGER_SECTIONS_IMPL=rust (default python); Python
# retains all INI writers.
# Built on bookworm (glibc 2.36); the runtime base below is Debian bookworm or newer, and glibc
# is backward compatible, so the dynamically linked binary runs there.
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS gguf-meta
# Merged Rust runtime dependency; checksum verified below.
ARG NOEVIA_RS_REF=6b15c0150ba2615a7d00721bb2b84cd0c676f8f9
ARG NOEVIA_RS_SHA256=5884831a7b5206c9c7d29321fdcf1b9b1e1928c106e34db12e54837c6eee9b0c
# The source's rust-toolchain.toml names 1.99.0 plus clippy/rustfmt; pin to the image's own
# toolchain so the build never downloads components it does not need.
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && cargo build --release --locked -p gguf-meta -p model-files-cli -p model-autoconfig-cli -p noevia-model-manager --features noevia-model-manager/native-host,noevia-model-manager/native-sections,noevia-model-manager/native-prompts,noevia-model-manager/native-diagnose,noevia-model-manager/native-settings,noevia-model-manager/native-model-updates,noevia-model-manager/native-model-detail,noevia-model-manager/native-download-targets \
 && native_features=$(./target/release/noevia-model-manager --features) \
 && printf '%s\n' "$native_features" | grep -Fx native-logs >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-host >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-sections >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-prompts >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-diagnose >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-settings >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-model-updates >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-model-detail >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-download-targets >/dev/null \
 && ./target/release/gguf-meta --help 2>&1 | grep -q usage \
 && ./target/release/model-files --help 2>&1 | grep -q usage \
 && ./target/release/model-autoconfig --help 2>&1 | grep -q usage

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
# noevia#1071: the native parent is permanent. Binaries before noevia-rs#142 still read this retired
# selector (their default was a Python-only mode), so the image pins it; newer binaries ignore it.
ENV MODEL_MANAGER_IMPL=rust
# #1334: native section reads stay off by default (python answers them).
# Compose must pass ${MODEL_MANAGER_SECTIONS_IMPL:-python}, never an empty value.
ENV MODEL_MANAGER_SECTIONS_IMPL=python
# #1318: native GET /api/v1/settings stays off by default; PUT always stays Python (single writer).
ENV MODEL_MANAGER_SETTINGS_IMPL=python
# #1318: native GET /api/v1/models/updates stays off by default; every update_check writer stays Python.
ENV MODEL_MANAGER_MODEL_UPDATES_IMPL=python
# #1318: native GET /api/v1/models/detail stays off by default; rust also needs TZ=UTC, and the live
# loadedOn/fit fields always come from this Python app through its private runtime call.
ENV MODEL_MANAGER_MODEL_DETAIL_IMPL=python
# #1318: native GET /api/v1/download-targets stays off by default; POST /api/v1/downloads stays Python.
ENV MODEL_MANAGER_DOWNLOAD_TARGETS_IMPL=python
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=gguf-meta /src/target/release/gguf-meta /usr/local/bin/gguf-meta
COPY --from=gguf-meta /src/target/release/model-files /usr/local/bin/model-files
COPY --from=gguf-meta /src/target/release/model-autoconfig /usr/local/bin/model-autoconfig
COPY --from=gguf-meta /src/target/release/noevia-model-manager /usr/local/bin/noevia-model-manager
COPY LICENSE ./
COPY app ./app
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8090/api/v1/health',timeout=4)"
CMD ["noevia-model-manager"]

# Qualification reuses the complete production image and actual app. The fixed
# Rust fixture launcher suppresses startup hooks; no engines or real model data.
FROM gguf-meta AS qualification-builder
COPY qualification/main.rs /tmp/sections-image-probe.rs
RUN rustc --edition=2021 /tmp/sections-image-probe.rs -o /tmp/sections-image-probe \
 && cargo build --release --locked -p noevia-model-manager --features native-host

COPY qualification/prompts.rs qualification/prompts-image.rs /tmp/prompts-qualification/
RUN rustc --edition=2021 /tmp/prompts-qualification/prompts-image.rs -o /tmp/prompts-image-probe

COPY qualification/diagnosis-image.rs /tmp/prompts-qualification/diagnosis-image.rs
RUN rustc --edition=2021 /tmp/prompts-qualification/diagnosis-image.rs -o /tmp/diagnosis-image-probe

FROM runtime AS qualification
COPY --from=qualification-builder /tmp/sections-image-probe /usr/local/bin/sections-image-probe
COPY --from=qualification-builder /tmp/prompts-image-probe /usr/local/bin/prompts-image-probe
COPY --from=qualification-builder /tmp/diagnosis-image-probe /usr/local/bin/diagnosis-image-probe
COPY --from=qualification-builder /src/target/release/noevia-model-manager /usr/local/bin/model-manager-sections-omission
COPY --from=qualification-builder /src/target/release/noevia-model-manager /usr/local/bin/model-manager-prompts-omission

COPY --from=qualification-builder /src/target/release/noevia-model-manager /usr/local/bin/model-manager-diagnosis-omission

# Default builds retain the production runtime and entrypoint.
FROM runtime AS production
