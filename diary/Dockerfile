# The Diary Rust leaves from sbstndalton/noevia-rs, built into one /out/ bundle (noevia#1071):
# tenant-assertion is the only tenant signature check (TENANT_ASSERTION_IMPL is retired, noevia#1071);
# workspace-restore-verify is behind WORKSPACE_RESTORE_VERIFY_IMPL. The six leaves below it ship for the
# Python adapters that already call them, each behind its own selector, all defaulting to python:
# diary-corpus-write-cli (DIARY_CORPUS_WRITE_IMPL), diary-corpus-parse (DIARY_CORPUS_PARSE_IMPL),
# diary-score-gap (DIARY_SCORE_GAP_IMPL), diary-temporal-cli (DIARY_TEMPORAL_FOCUS_IMPL),
# diary-workspace-export (WORKSPACE_EXPORT_IMPL) and diary-workspace-import-apply (WORKSPACE_IMPORT_APPLY_IMPL).
# noevia-diary (D4-D5) is the Rust Diary server process that Python starts on 127.0.0.1:8011 only when
# DIARY_RUST_ROUTES names a route or DIARY_RUST_WRITES a tenant. Both are empty in this image, so the process
# is never started and behaviour is unchanged until the owner sets them.
# Pinned to release/versions.lock's NOEVIA_RS_REF like the OCR and model-manager
# images, fetched as an anonymous tarball whose sha256 is checked before anything is built. Built on
# bookworm (glibc 2.36); the runtime base is the same or newer Debian. The release's Diary overlay
# (noevia deploy/examples/diary-overlay.sh) builds the same list from these ARG defaults into the same
# /out/ bundle, so an overlaid image carries the binaries at the same paths without a pip install.
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS tenant-assertion
# Merged decode-v1 Rust runtime dependency (noevia-rs#301); checksum verified below.
ARG NOEVIA_RS_REF=ec4036843f9f6143f3006e5a706c4f8c1be67fa4
ARG NOEVIA_RS_SHA256=19c5773e07d905076ffc22e526a1b584d27f2b6248a2877cf1184ea780308e50
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never CARGO_BUILD_JOBS=2
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && cargo build --release --locked -p tenant-assertion-cli -p workspace-restore-verify-cli \
      -p diary-corpus-write-cli -p diary-corpus-parse -p diary-score-gap -p diary-temporal-focus \
      -p diary-workspace-export -p diary-workspace-import-apply \
      -p diary-entity-focus -p diary-index-transform -p diary-storage-backoff -p diary-workspace-rules -p diary-backup-format -p diary-backup-run -p diary-operator \
      -p noevia-diary \
 && ./target/release/tenant-assertion self-test | grep -qx ok \
 && ./target/release/workspace-restore-verify self-test | grep -qx ok \
 && mkdir /out \
 && cd target/release \
 && cp tenant-assertion workspace-restore-verify diary-corpus-write-cli diary-corpus-parse diary-score-gap \
      diary-temporal-cli diary-workspace-export diary-workspace-import-apply \
      diary-entity-focus diary-index-transform diary-storage-backoff diary-workspace-rules diary-backup-format diary-backup-run diary-backup-restore diary-config noevia-diary /out/ \
 && cd /src \
 && rm -rf /src/target /usr/local/cargo/registry

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade "pip>=26.2" && pip install --no-cache-dir -r requirements.txt

COPY --from=tenant-assertion /out/ /usr/local/bin/
RUN tenant-assertion self-test | grep -qx ok && workspace-restore-verify self-test | grep -qx ok
# The native Diary binaries at the fixed paths the Python code already defaults to (agent/rust_forward.py
# DEFAULT_BIN, agent/tenant_assertion.py DEFAULT_BIN, agent/workspace_export_native.py BINARY). These three are
# the same values, stated once so an operator can see them; the service falls back to the same paths if they are
# unset. DIARY_WORKSPACE_EXPORT_BIN is read by the D4b tests only. DIARY_RUST_ROUTES and DIARY_RUST_WRITES are
# deliberately NOT set here: empty means the Rust server is never started (see agent/rust_forward.py).
ENV DIARY_RUST_SERVER_BIN=/usr/local/bin/noevia-diary \
    DIARY_WORKSPACE_EXPORT_BIN=/usr/local/bin/diary-workspace-export \
    TENANT_ASSERTION_BIN=/usr/local/bin/tenant-assertion
RUN test -x "$DIARY_RUST_SERVER_BIN" && test -x "$DIARY_WORKSPACE_EXPORT_BIN" && test -x "$TENANT_ASSERTION_BIN"
COPY agent/ ./agent/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY entrypoint.sh /usr/local/bin/diary-entrypoint
RUN chmod 755 /usr/local/bin/diary-entrypoint \
 && diary-backup-restore --help | grep -q '^usage: backup_restore.py ' \
 && diary-config /nonexistent/diary-config-smoke.yaml | grep -qx '{"configured":true}'
# Prove real installed binaries and historical Python spellings offline, including
# credential-canary checks. Build args bind this candidate to a verified Rust archive.
RUN mkdir /operator-smoke-evidence \
 && DIARY_OPERATOR_SMOKE_ROOT=/operator-smoke-evidence python scripts/smoke_operator_image.py \
 && rm -rf /operator-smoke-evidence
# Prove the installed binary speaks decode-v1, including payload/version/EOF checks.
RUN python scripts/smoke_workspace_decode.py
# Every other leaf answers through its real adapter and agrees with Python on synthetic data. The selectors
# are set only inside this smoke; the service keeps every one at python. The overlay runs the same script.
RUN python scripts/smoke_diary_leaves.py --all

# Bind-mounted state must be writable by the image's runtime user.

EXPOSE 8010
# Authed probe: /api/health returns 401 once DIARY_AUTH_TOKEN is set (same fix as
# the compose healthcheck and the cowork sidecar's override). Exec form - Docker
# runs the JSON array directly, no shell, so the python -c string needs only one
# quoting level; an uncaught 401 HTTPError exits non-zero (no `|| exit 1` needed).
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request; h={'Authorization':'Bearer '+os.environ.get('DIARY_AUTH_TOKEN','')} if os.environ.get('DIARY_AUTH_TOKEN') else {}; print(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8010/api/health', headers=h), timeout=5).status)"]

ENTRYPOINT ["diary-entrypoint"]
CMD ["uvicorn", "agent.app:app", "--host", "0.0.0.0", "--port", "8010"]
