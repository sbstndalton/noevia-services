# syntax=docker/dockerfile:1.7
# noevia model manager: Model Loader (scratchhax/model-loader, MIT) folded into noevia.
# JSON API only (/api/v1); noevia web renders every screen.

# gguf-meta (#909) and model-files (#964): bounded Rust leaves from sbstndalton/noevia-rs,
# pinned to a merged NOEVIA_RS_REF and fetched as an anonymous tarball whose
# sha256 is checked before anything is built. gguf-meta is used only when GGUF_PARSER=rust,
# model-files only when MODEL_FILES_IMPL=rust (both default python; since #1021 also models.ini backups),
# model-autoconfig only when MODEL_AUTOCONFIG=rust (default python; it only ever confirms the Python answer).
# Native health/logs/host (#1319, #1330): MODEL_MANAGER_IMPL defaults to python; the entrypoint
# execs the unchanged uvicorn command unless explicitly selected as rust.
# Native section GET reads require MODEL_MANAGER_SECTIONS_IMPL=rust and the
# Rust manager. Both selectors default to python; Python retains all INI writers.
# Built on bookworm (glibc 2.36); the runtime base below is Debian bookworm or newer, and glibc
# is backward compatible, so the dynamically linked binary runs there.
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS gguf-meta
# Merged Rust runtime dependency; checksum verified below.
ARG NOEVIA_RS_REF=f5d627aaafd0d381872c3d2ddc8137d2f10383a7
ARG NOEVIA_RS_SHA256=796a00832e42b54b97687e2a467ecd0405f9c59c7361a9d40b37ba17701ffe4e
# The source's rust-toolchain.toml names 1.99.0 plus clippy/rustfmt; pin to the image's own
# toolchain so the build never downloads components it does not need.
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && cargo build --release --locked -p gguf-meta -p model-files-cli -p model-autoconfig-cli -p noevia-model-manager --features noevia-model-manager/native-host,noevia-model-manager/native-sections,noevia-model-manager/native-prompts,noevia-model-manager/native-diagnose \
 && native_features=$(./target/release/noevia-model-manager --features) \
 && printf '%s\n' "$native_features" | grep -Fx native-logs >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-host >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-sections >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-prompts >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-diagnose >/dev/null \
 && ./target/release/gguf-meta --help 2>&1 | grep -q usage \
 && ./target/release/model-files --help 2>&1 | grep -q usage \
 && ./target/release/model-autoconfig --help 2>&1 | grep -q usage

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
# #1334: native section reads stay off by default (python answers them); rust needs MODEL_MANAGER_IMPL=rust
# too and is refused otherwise. Compose must pass ${MODEL_MANAGER_SECTIONS_IMPL:-python}, never an empty value.
ENV MODEL_MANAGER_SECTIONS_IMPL=python
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
