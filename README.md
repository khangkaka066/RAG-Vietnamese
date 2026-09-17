# Vietnamese RAG & LLM Evaluation Service

Status: **In Progress**

A production-oriented portfolio project for the VSF AI Engineer role. It is a small, provider-agnostic service for Vietnamese document retrieval, grounded answers, citations, tool calls, and repeatable evaluation.

## Why this project

This project targets the main capabilities requested in the job description:

- NLP and Vietnamese text understanding
- RAG-style retrieval and grounded responses
- Agent/tool-use architecture
- FastAPI inference API
- JSONL evaluation data and measurable quality gates
- Docker, pytest, and CI-ready project structure

## Architecture

```text
Documents (JSONL)
        |
        v
Document loader -> Vietnamese lexical retriever -> Context selector
                                                        |
                                                        v
                               Extractive answer generator + citations
                                                        |
                                                        v
                                      FastAPI /query and /evaluate

User query -> Tool router -> registered domain tool (when applicable)
```

The current implementation is a deterministic baseline. A future milestone will add an interchangeable LLM provider, embeddings, reranking, and online experiment tracking without changing the API contract.

## Quick start

```bash
cd vsf-vietnamese-rag-evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
uvicorn vsf_rag.api:app --app-dir src --reload
```

Open the interactive API documentation at `http://localhost:8000/docs`.

Example request:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"RAG cần đánh giá bằng những chỉ số nào?","top_k":3}'
```

Run the evaluation set:

```bash
python scripts/run_eval.py
```

## Docker

```bash
docker build -t vietnamese-rag-eval .
docker run --rm -p 8000:8000 vietnamese-rag-eval
```

## Current quality gates

- Retrieval hit rate on the checked-in evaluation set
- Answer keyword coverage
- Citation coverage
- Unavailable behavior when evidence is insufficient
- Unit tests for tokenization, ranking, evaluation, and failure handling

## Roadmap

1. Add multilingual embedding retrieval and a cross-encoder reranker.
2. Add an LLM provider interface with local-model and API-backed implementations.
3. Add Vietnamese RAG faithfulness and answer-relevance evaluation.
4. Add MLflow/W&B experiment tracking and latency metrics.
5. Add a tool-using agent with explicit planning, tool-call validation, and trace logging.
6. Add GitHub Actions, OpenAPI examples, and a small deployed demo.
