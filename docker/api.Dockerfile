FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    DATA_DIR=/app/data

WORKDIR /app

# CPU-only PyTorch keeps the image a fraction of the size of the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache
COPY --chown=app:app backend backend
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
