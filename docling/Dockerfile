FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch, explicitly. On Linux, `pip install torch` from PyPI resolves
# to the CUDA build and drags in the nvidia-* wheels — multiple GB of CUDA
# runtime that this image will never execute, on a host whose GPU is committed
# to llama.cpp. The CPU index is the difference between a ~1 GB layer and a
# ~4 GB one.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision

# Narrow extras rather than `docling[all]`:
#   convert-core / models-local  — the layout + TableFormer pipeline
#   format-*                     — PDF, Office, OpenDocument, HTML
#   feat-ocr-tesserocr           — reuse the tesseract already installed above
# Deliberately absent: feat-ocr-easyocr (~13 s/page on CPU) and
# models-vlm-inline (the 2B chart model). Charts are a separate decision.
RUN pip install --no-cache-dir \
      "docling-slim[convert-core,models-local,format-pdf,format-office,format-opendocument,format-html,feat-ocr-tesserocr]==2.129.0"

# TableFormer imports cv2 at construction time (docling_ibm_models'
# tf_predictor), and none of the extras above pull OpenCV in — so without this
# the converter raises ModuleNotFoundError the moment do_table_structure is on,
# which is the feature this service exists for. Headless: the GUI build wants
# libGL and other X libraries that are not in the slim base image.
RUN pip install --no-cache-dir "opencv-python-headless<5"

# Fetch the models at BUILD time, into the image. The running container then
# never reaches HuggingFace — which matters because this host is meant to work
# without outbound access, and because a first upload should not stall behind a
# 1.6 GB download. Also means the image is a fixed, auditable artefact.
ENV HF_HUB_OFFLINE=1 \
    DOCLING_ARTIFACTS_PATH=/opt/docling-models \
    OMP_THREAD_LIMIT=1 \
    OMP_NUM_THREADS=2
# Name the models explicitly. With no arguments `docling-tools models download`
# resolves its whole default set, which includes the PP-OCR/rapidocr
# recognizers — and rapidocr is not installed here, so resolution raises
# ImportError and the build dies. Naming layout+tableformer is also what this
# pipeline actually uses: OCR goes through the Tesseract CLI, which needs no
# downloaded model at all. Measured 669 MB, not the ~1.6 GB first assumed.
RUN HF_HUB_OFFLINE=0 docling-tools models download -o /opt/docling-models \
      layout tableformer \
    && chmod -R a+rX /opt/docling-models

WORKDIR /app
COPY server.py extract.py selftest.py ./
USER 65534:65534
EXPOSE 8031
HEALTHCHECK --interval=30s --timeout=6s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8031/health',timeout=5)"
CMD ["python", "server.py"]
