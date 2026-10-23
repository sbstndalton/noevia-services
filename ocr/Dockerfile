# Native OCR (#1071): noevia-ocr from sbstndalton/noevia-rs is the only owner of every OCR route
# (it reads DOCX through the docx-text library). It is pinned to release/versions.lock's
# NOEVIA_RS_REF and fetched as an anonymous tarball whose sha256 is checked before anything is
# built. Built on bookworm (glibc 2.36), the same Debian as the runtime base below. The retired
# Python server is the test oracle at services bbed9b8, never part of this image.
FROM rust:1.99-slim-bookworm@sha256:2c3a22f0a5533ea2dd5a16627bc841228151faa2d4de2644ac9987e4a2f1f2fa AS native
# Merged Rust runtime dependency; checksum verified below.
ARG NOEVIA_RS_REF=d3a0a9c21ae7c1dc5ea068fbd8dd33ff74370695
ARG NOEVIA_RS_SHA256=55a2f5b7034f1ee5a044c24cfe6715265bfcbd64c88caf9471e65fd96ed56590
# The source's rust-toolchain.toml names 1.99.0 plus clippy/rustfmt; pin to the image's own
# toolchain so the build never downloads components it does not need.
ENV RUSTUP_TOOLCHAIN=1.99.0 CARGO_TERM_COLOR=never
WORKDIR /src
ADD https://codeload.github.com/sbstndalton/noevia-rs/tar.gz/${NOEVIA_RS_REF} /tmp/noevia-rs.tar.gz
RUN echo "${NOEVIA_RS_SHA256}  /tmp/noevia-rs.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/noevia-rs.tar.gz --strip-components=1 \
 && rm /tmp/noevia-rs.tar.gz \
 && cargo build --release --locked -p noevia-ocr --features native-ocr

# No interpreter: the native server and the external poppler, tesseract (eng+deu) and ghostscript
# engines only.
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgcc-s1 ghostscript poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
 && rm -rf /var/lib/apt/lists/*
COPY --from=native /src/target/release/noevia-ocr /usr/local/bin/noevia-ocr
RUN noevia-ocr --features > /tmp/ocr-features && grep -qx native-ocr /tmp/ocr-features && rm /tmp/ocr-features
USER 65534:65534
ENV OMP_THREAD_LIMIT=1
EXPOSE 8030
HEALTHCHECK --interval=30s --timeout=6s --start-period=10s --retries=3 CMD ["/usr/local/bin/noevia-ocr", "--healthcheck"]
ENTRYPOINT ["/usr/local/bin/noevia-ocr"]
