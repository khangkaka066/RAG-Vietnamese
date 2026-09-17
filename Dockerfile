FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "vsf_rag.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
