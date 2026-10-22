# syntax=docker/dockerfile:1.7
# noevia model manager: Model Loader (scratchhax/model-loader, MIT) folded into noevia.
# JSON API only (/api/v1); noevia web renders every screen.

# The merged, checksum-verified Rust source supplies the sole production service.
# All model-manager HTTP routes, stores, jobs and healthchecks are native (#1318/#1071).
# Python and the previous supervisor binary exist only in the qualification targets.
# The bookworm builder and runtime use compatible glibc versions.
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS gguf-meta
# Merged Rust runtime dependency; checksum verified below. The inference gate (noevia#2115)
# is built from the same source in its own cargo invocation, so its dependencies' features never
# unify into the model manager's build.
ARG NOEVIA_RS_REF=bac2f3e1dd56b079cd02d2be3333bc7b6db2e12d
ARG NOEVIA_RS_SHA256=d059969eba60f0993cd64655e539edb93fc323a476ef09a310f8b92686db6281
# The source's rust-toolchain.toml names 1.99.0 plus clippy/rustfmt; pin to the image's own
# toolchain so the build never downloads components it does not need.
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never CARGO_BUILD_JOBS=8
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && cargo build --release --locked -p gguf-meta -p model-files-cli -p model-autoconfig-cli -p noevia-model-manager --features noevia-model-manager/native-host,noevia-model-manager/native-sections,noevia-model-manager/native-prompts,noevia-model-manager/native-diagnose,noevia-model-manager/native-settings,noevia-model-manager/native-model-updates,noevia-model-manager/native-model-detail,noevia-model-manager/native-download-targets,noevia-model-manager/native-badges,noevia-model-manager/native-bench-runs,noevia-model-manager/native-section-writes,noevia-model-manager/native-settings-write,noevia-model-manager/native-backend-control,noevia-model-manager/native-benchmark-overview,noevia-model-manager/native-check-updates,noevia-model-manager/native-benchmark-runner,noevia-model-manager/native-downloads,noevia-model-manager/native-model-delete,noevia-model-manager/native-draft-heads,noevia-model-manager/native-backends,noevia-model-manager/native-model-list,noevia-model-manager/native-search,noevia-model-manager/native-section-autoconfig,noevia-model-manager/native-telemetry,noevia-model-manager/native-bench-observe,noevia-model-manager/native-db-init,noevia-model-manager/native-token-test,noevia-model-manager/native-safe-defaults,noevia-model-manager/native-download-plan,noevia-model-manager/native-routing \
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
 && printf '%s\n' "$native_features" | grep -Fx native-badges >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-bench-runs >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-section-writes >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-settings-write >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-backend-control >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-benchmark-overview >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-check-updates >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-benchmark-runner >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-downloads >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-model-delete >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-draft-heads >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-backends >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-model-list >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-search >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-section-autoconfig >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-telemetry >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-bench-observe >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-db-init >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-token-test >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-safe-defaults >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-download-plan >/dev/null \
 && printf '%s\n' "$native_features" | grep -Fx native-routing >/dev/null \
 && ./target/release/gguf-meta --help 2>&1 | grep -q usage \
 && ./target/release/model-files --help 2>&1 | grep -q usage \
 && ./target/release/model-autoconfig --help 2>&1 | grep -q usage \
 && cargo build --release --locked -p noevia-inference-gate \
 && { env -u GATE_UPSTREAM ./target/release/noevia-inference-gate 2>&1 || true; } | grep -q "GATE_UPSTREAM is required"

# Test-only frozen supervisor oracle. It must never supply the production binary.
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS legacy-builder
ARG LEGACY_ORACLE_REF=76a8e682a8cb0f02f9df5363281bbe0ede381d5b
ARG LEGACY_ORACLE_SHA256=f2725e2a564b0a920ac5645ed81d2427b08d927749a0e842e5784e4996a6dd5c
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never CARGO_BUILD_JOBS=8
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${LEGACY_ORACLE_REF} /tmp/legacy-oracle.tar.gz
RUN echo "${LEGACY_ORACLE_SHA256}  /tmp/legacy-oracle.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/legacy-oracle.tar.gz --strip-components=1 \
 && rm /tmp/legacy-oracle.tar.gz \
 && cargo build --release --locked -p gguf-meta -p model-files-cli -p model-autoconfig-cli -p noevia-model-manager --features noevia-model-manager/native-host,noevia-model-manager/native-sections,noevia-model-manager/native-prompts,noevia-model-manager/native-diagnose

FROM python:3.12-slim AS legacy-runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
ENV MODEL_MANAGER_SECTIONS_IMPL=python MODEL_MANAGER_SETTINGS_IMPL=python
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=legacy-builder /src/target/release/gguf-meta /usr/local/bin/gguf-meta
COPY --from=legacy-builder /src/target/release/model-files /usr/local/bin/model-files
COPY --from=legacy-builder /src/target/release/model-autoconfig /usr/local/bin/model-autoconfig
COPY --from=legacy-builder /src/target/release/noevia-model-manager /usr/local/bin/noevia-model-manager
COPY LICENSE ./
COPY app ./app
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8090/api/v1/health',timeout=4)"
CMD ["noevia-model-manager"]

# Frozen supervisor qualification: actual Python remains a test oracle, never a
# shipping dependency. The fixed launcher suppresses hardware/engine startup.
FROM legacy-builder AS qualification-builder
COPY qualification/main.rs /tmp/sections-image-probe.rs
RUN rustc --edition=2021 /tmp/sections-image-probe.rs -o /tmp/sections-image-probe \
 && cargo build --release --locked -p noevia-model-manager --features native-host

COPY qualification/prompts.rs qualification/prompts-image.rs /tmp/prompts-qualification/
RUN rustc --edition=2021 /tmp/prompts-qualification/prompts-image.rs -o /tmp/prompts-image-probe

COPY qualification/diagnosis-image.rs /tmp/prompts-qualification/diagnosis-image.rs
RUN rustc --edition=2021 /tmp/prompts-qualification/diagnosis-image.rs -o /tmp/diagnosis-image-probe

FROM legacy-runtime AS qualification
COPY --from=qualification-builder /tmp/sections-image-probe /usr/local/bin/sections-image-probe
COPY --from=qualification-builder /tmp/prompts-image-probe /usr/local/bin/prompts-image-probe
COPY --from=qualification-builder /tmp/diagnosis-image-probe /usr/local/bin/diagnosis-image-probe
COPY --from=qualification-builder /src/target/release/noevia-model-manager /usr/local/bin/model-manager-sections-omission
COPY --from=qualification-builder /src/target/release/noevia-model-manager /usr/local/bin/model-manager-prompts-omission

COPY --from=qualification-builder /src/target/release/noevia-model-manager /usr/local/bin/model-manager-diagnosis-omission

# Production has no Python, app source, uvicorn or helper executable to fall back to.
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587 AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgcc-s1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY --from=gguf-meta /src/target/release/noevia-model-manager /usr/local/bin/noevia-model-manager
# The inference gate runs from this image as its own compose service (command override);
# the image's default process stays the model manager.
COPY --from=gguf-meta /src/target/release/noevia-inference-gate /usr/local/bin/noevia-inference-gate
COPY LICENSE ./
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["noevia-model-manager", "--healthcheck"]
CMD ["noevia-model-manager"]

# Compile the exact shipping-process contract against the same verified source.
FROM gguf-meta AS native-qualification-builder
RUN set -eu; \
    cargo test --release --locked -p noevia-model-manager --features native-routing --test native_service --no-run; \
    n=0; for artifact in target/release/deps/native_service-*; do \
      if [ -f "$artifact" ] && [ -x "$artifact" ]; then \
        cp "$artifact" /tmp/native-service-test; n=$((n+1)); \
      fi; \
    done; test "$n" = 1

# Actual production rootfs plus an isolated Rust test driver; no Python oracle.
FROM runtime AS native-qualification
HEALTHCHECK NONE
COPY --from=native-qualification-builder /tmp/native-service-test /usr/local/bin/native-service-test
ENV MODEL_MANAGER_PRODUCTION_BIN=/usr/local/bin/noevia-model-manager
CMD ["native-service-test", "--test-threads=1"]

# Default builds always select the native production runtime.
FROM runtime AS production
