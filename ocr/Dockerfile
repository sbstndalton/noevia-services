FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ghostscript poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server.py docx_text.py pdf_reduce.py ./
USER 65534:65534
ENV OMP_THREAD_LIMIT=1
EXPOSE 8030
HEALTHCHECK --interval=30s --timeout=6s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8030/health',timeout=5)"
CMD ["python", "server.py"]
