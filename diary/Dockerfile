# tenant-assertion: the bounded Rust leaf from sbstndalton/noevia-rs behind TENANT_ASSERTION_IMPL
# (default python), pinned to release/versions.lock's NOEVIA_RS_REF like the OCR and model-manager
# images, fetched as an anonymous tarball whose sha256 is checked before anything is built. Built on
# bookworm (glibc 2.36); the runtime base is the same or newer Debian. The release's Diary overlay
# (noevia deploy/examples/diary-overlay.sh) builds this same stage from these ARG defaults, so an
# overlaid image carries the binary at the same path without a pip install.
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS tenant-assertion
ARG NOEVIA_RS_REF=be174b12559fd09c1f38a97ae155d353f3e8f77b
ARG NOEVIA_RS_SHA256=561ee90ffcac741818584f4efa7659feafef3bc54ac8d1764694e729289fce0b
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && cargo build --release --locked -p tenant-assertion-cli \
 && ./target/release/tenant-assertion self-test | grep -qx ok

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade "pip>=26.2" && pip install --no-cache-dir -r requirements.txt

COPY --from=tenant-assertion /src/target/release/tenant-assertion /usr/local/bin/tenant-assertion
RUN tenant-assertion self-test | grep -qx ok
COPY agent/ ./agent/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Bind-mounted state must be writable by the image's runtime user.

EXPOSE 8010
# Authed probe: /api/health returns 401 once DIARY_AUTH_TOKEN is set (same fix as
# the compose healthcheck and the cowork sidecar's override). Exec form — Docker
# runs the JSON array directly, no shell, so the python -c string needs only one
# quoting level; an uncaught 401 HTTPError exits non-zero (no `|| exit 1` needed).
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request; h={'Authorization':'Bearer '+os.environ.get('DIARY_AUTH_TOKEN','')} if os.environ.get('DIARY_AUTH_TOKEN') else {}; print(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8010/api/health', headers=h), timeout=5).status)"]

CMD ["uvicorn", "agent.app:app", "--host", "0.0.0.0", "--port", "8010"]
