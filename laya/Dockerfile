# syntax=docker/dockerfile:1.7
# This global argument remains visible to the final production FROM.
ARG RUNTIME_IMAGE=python:3.12-slim
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS native-builder
ARG NOEVIA_RS_REF=14fb71febfcd5957cdb376ce3c9883f5e91ad9c6
ARG NOEVIA_RS_SHA256=d883b21abc513c71d9619e48e5bf6ad403baa06a91036262d801677e75d1806c
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && CARGO_TARGET_DIR=/native cargo build --release --locked -p noevia-laya --features native-laya \
 && CARGO_TARGET_DIR=/featureless cargo build --release --locked -p noevia-laya
COPY qualification/Cargo.toml qualification/Cargo.lock /qualification/
COPY qualification/src /qualification/src
RUN cargo build --release --locked --manifest-path /qualification/Cargo.toml --target-dir /probe

# CI builds only these lightweight targets: no engine pip layer or model assets.
FROM python:3.12-slim AS synthetic-base
WORKDIR /service
COPY server.py /service/server.py
COPY --from=native-builder /probe/release/laya-image-probe /usr/local/bin/laya-image-probe
ENV USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
USER 1000:1000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD ["laya-image-probe", "health"]
FROM synthetic-base AS synthetic
COPY --from=native-builder /native/release/noevia-laya /usr/local/bin/noevia-laya
ENTRYPOINT ["noevia-laya"]
FROM synthetic-base AS synthetic-featureless
COPY --from=native-builder /featureless/release/noevia-laya /usr/local/bin/noevia-laya
ENTRYPOINT ["noevia-laya"]

# Production engine installation instructions below are byte-frozen from4f3.
# CI must never build/start this target; model provisioning is a separate action.
FROM ${RUNTIME_IMAGE}
USER root
ARG REUSE_CPU_RUNTIME=false
RUN if [ "$REUSE_CPU_RUNTIME" != true ]; then pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir transformers==5.17.0 huggingface_hub==1.32.0 safetensors==0.8.0 numpy==2.5.2; fi
RUN pip install --no-cache-dir --no-deps laya==0.3.5
WORKDIR /service
COPY server.py /service/server.py
COPY --from=native-builder /native/release/noevia-laya /usr/local/bin/noevia-laya
ENV USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
USER 1000:1000
ENTRYPOINT ["noevia-laya"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8040/health',timeout=3)"
