# Single-stage build: the default (and CI/Docker) retriever/generator config
# is fully offline (lexical retrieval + extractive fallback, see README), so
# no heavy torch/sentence-transformers dependency is required by default.
# Pass `--build-arg EXTRAS='[embeddings]'` to build a variant with the
# optional embedding retriever included.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts

ARG EXTRAS=""
RUN pip install --no-cache-dir ".${EXTRAS}"

RUN useradd -m appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "vsf_rag.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
