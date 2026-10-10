# syntax=docker/dockerfile:1.7
# noevia model manager: Model Loader (scratchhax/model-loader, MIT) folded into noevia.
# JSON API only (/api/v1); noevia web renders every screen.

# gguf-meta (#909) and model-files (#964): bounded Rust leaves from sbstndalton/noevia-rs,
# pinned to release/versions.lock's NOEVIA_RS_REF and fetched as an anonymous tarball whose
# sha256 is checked before anything is built. gguf-meta is used only when GGUF_PARSER=rust,
# model-files only when MODEL_FILES_IMPL=rust (both default python; since #1021 also models.ini backups),
# model-autoconfig only when MODEL_AUTOCONFIG=rust (default python; it only ever confirms the Python answer).
# Built on bookworm (glibc 2.36); the runtime base below is Debian bookworm or newer, and glibc
# is backward compatible, so the dynamically linked binary runs there.
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS gguf-meta
ARG NOEVIA_RS_REF=68c9f14efb2dfe7751431080ce3c021a6bd49439
ARG NOEVIA_RS_SHA256=585cfb115ca68072d2b110d852b50d1a99cc35d35abc1252f8f4c74ac9441e9f
# The source's rust-toolchain.toml names 1.99.0 plus clippy/rustfmt; pin to the image's own
# toolchain so the build never downloads components it does not need.
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && cargo build --release --locked -p gguf-meta -p model-files-cli -p model-autoconfig-cli \
 && ./target/release/gguf-meta --help 2>&1 | grep -q usage \
 && ./target/release/model-files --help 2>&1 | grep -q usage \
 && ./target/release/model-autoconfig --help 2>&1 | grep -q usage

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=gguf-meta /src/target/release/gguf-meta /usr/local/bin/gguf-meta
COPY --from=gguf-meta /src/target/release/model-files /usr/local/bin/model-files
COPY --from=gguf-meta /src/target/release/model-autoconfig /usr/local/bin/model-autoconfig
COPY LICENSE ./
COPY app ./app
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8090/api/v1/health',timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
