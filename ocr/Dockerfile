# docx-text (#981): a bounded Rust leaf from sbstndalton/noevia-rs, pinned to
# release/versions.lock's NOEVIA_RS_REF and fetched as an anonymous tarball whose sha256 is checked
# before anything is built. Used only when DOCX_TEXT_IMPL=rust (default python). Built on bookworm
# (glibc 2.36), the same Debian as the runtime base below.
FROM rust:1.99-slim-bookworm AS docx-text
ARG NOEVIA_RS_REF=a6a50ad971339eea7534d9266d887e04a26af384
ARG NOEVIA_RS_SHA256=056ade791458364563b7711ccb53bb604dffff046da96740d4aaa607e5acc676
# The source's rust-toolchain.toml names 1.99.0 plus clippy/rustfmt; pin to the image's own
# toolchain so the build never downloads components it does not need.
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && cargo build --release --locked -p docx-text-cli \
 && ./target/release/docx-text --help 2>&1 | grep -q usage

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ghostscript poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu && rm -rf /var/lib/apt/lists/*
COPY --from=docx-text /src/target/release/docx-text /usr/local/bin/docx-text
# Smoke test on the runtime base: an empty ZIP (a bare end-of-central-directory record) is refused
# as having no main document, with exit 1 and nothing on stdout.
RUN printf 'PK\005\006\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000' > /tmp/empty.zip; \
    docx-text extract < /tmp/empty.zip > /tmp/out 2> /tmp/err; code=$?; \
    [ "$code" = 1 ] && [ ! -s /tmp/out ] && grep -qx 'docx-text: refused: missing' /tmp/err \
    && rm -f /tmp/empty.zip /tmp/out /tmp/err
WORKDIR /app
COPY server.py docx_text.py pdf_reduce.py ./
USER 65534:65534
ENV OMP_THREAD_LIMIT=1
EXPOSE 8030
HEALTHCHECK --interval=30s --timeout=6s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8030/health',timeout=5)"
CMD ["python", "server.py"]
