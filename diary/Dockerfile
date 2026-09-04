FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
