ARG RUNTIME_IMAGE=python:3.12-slim
FROM ${RUNTIME_IMAGE}
USER root
ARG REUSE_CPU_RUNTIME=false
RUN if [ "$REUSE_CPU_RUNTIME" != true ]; then pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir transformers==5.17.0 huggingface_hub==1.32.0 safetensors==0.8.0 numpy==2.5.2; fi
RUN pip install --no-cache-dir --no-deps laya==0.3.5
WORKDIR /service
COPY server.py /service/server.py
ENV USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
USER 1000:1000
ENTRYPOINT ["python", "/service/server.py"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8040/health',timeout=3)"
