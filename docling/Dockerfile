FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

# Reproducible build (#890). Every Python dependency is pinned by version and sha256, so a layer
# cache miss rebuilds the same image instead of re-resolving the newest torch and transitive
# dependencies. `pip install --no-deps --require-hashes` forbids pip from resolving anything.
# Normal docling releases do not rebuild at all: they ship as an app-only overlay FROM the previous
# cowork-docling image (docs/deployment.md, "Docling release path"). Rebuild this file only to
# change a dependency, and then regenerate the lock files first (headers say how).
COPY requirements.txt requirements-torch-cpu.txt download_models.py /build/

# CPU-only torch, explicitly. On Linux, `pip install torch` from PyPI resolves
# to the CUDA build and drags in the nvidia-* wheels — multiple GB of CUDA
# runtime that this image will never execute, on a host whose GPU is committed
# to llama.cpp. The CPU index is the difference between a ~1 GB layer and a
# ~4 GB one. torch and torchvision are locked in requirements-torch-cpu.txt.
RUN pip install --no-cache-dir --no-deps --require-hashes \
      --index-url https://download.pytorch.org/whl/cpu -r /build/requirements-torch-cpu.txt

# Everything else, from requirements.txt (generated from requirements.in):
#   docling-slim[convert-core,models-local,format-pdf,format-office,format-opendocument,
#                format-html,feat-ocr-tesserocr]==2.129.0 — narrow extras rather than `docling[all]`:
#     convert-core / models-local  — the layout + TableFormer pipeline
#     format-*                     — PDF, Office, OpenDocument, HTML
#     feat-ocr-tesserocr           — Tesseract OCR via tesserocr (its wheel bundles libtesseract;
#                                    the tesseract-ocr packages above provide the CLI and the data)
#   Deliberately absent: feat-ocr-easyocr (~13 s/page on CPU) and models-vlm-inline (the 2B chart
#   model). Charts are a separate decision.
#   opencv-python-headless — TableFormer imports cv2 at construction time (docling_ibm_models'
#   tf_predictor) and none of the extras pull OpenCV in; without it the converter raises
#   ModuleNotFoundError the moment do_table_structure is on, which is the feature this service
#   exists for. Headless: the GUI build wants libGL and X libraries absent from the slim base.
RUN pip install --no-cache-dir --no-deps --require-hashes -r /build/requirements.txt

# Fetch the models at BUILD time, into the image. The running container then
# never reaches HuggingFace — which matters because this host is meant to work
# without outbound access, and because a first upload should not stall behind a
# 1.6 GB download. Also means the image is a fixed, auditable artefact.
ENV HF_HUB_OFFLINE=1 \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models \
    OMP_THREAD_LIMIT=1 \
    OMP_NUM_THREADS=2
# Only `layout` and `tableformer` are fetched, and pinned to exact Hugging Face commits by
# download_models.py. `docling-tools models download` with no arguments resolves its whole default
# set, which includes the PP-OCR/rapidocr recognizers — and rapidocr is not installed here, so
# resolution raises ImportError and the build dies. Even with the two names it resolves the layout
# model at the floating revision "main", which is why this does not call it. OCR goes through the
# Tesseract CLI, which needs no downloaded model at all. Measured 669 MB, not the ~1.6 GB first
# assumed.
RUN HF_HUB_OFFLINE=0 python /build/download_models.py /opt/docling-models \
    && chmod -R a+rX /opt/docling-models && rm -rf /build

WORKDIR /app
# isolation.py + isolated_worker.py run each conversion in a child process the server can kill
# on a deadline or client disconnect (#854); the server cannot start without them.
# synthetic_pdfs.py builds the #700 fixtures so selftest.py can be pointed at a
# known text-over-picture page on the host without any real document.
COPY server.py extract.py isolation.py isolated_worker.py selftest.py synthetic_pdfs.py ./
USER 65534:65534
EXPOSE 8031
HEALTHCHECK --interval=30s --timeout=6s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8031/health',timeout=5)"
CMD ["python", "server.py"]
