# Vietnamese RAG & LLM Evaluation Service

Status: **In Progress**

A small, provider-agnostic service for Vietnamese document retrieval, grounded answers, citations, tool calls, and repeatable evaluation.

## Ứng dụng thực tế

Sau khi hoàn thiện, kiến trúc này có thể áp dụng cho các bài toán cần tra cứu
và trả lời dựa trên kho tài liệu tiếng Việt nội bộ, ví dụ:

- Trợ lý tra cứu chính sách/quy trình nội bộ doanh nghiệp (HR, pháp lý, vận hành)
- Chatbot hỗ trợ khách hàng dựa trên tài liệu sản phẩm/FAQ, có trích dẫn nguồn
  để kiểm chứng thay vì trả lời "ảo giác"
- Công cụ tìm kiếm ngữ nghĩa cho tài liệu kỹ thuật, hợp đồng, hoặc cơ sở tri
  thức nội bộ khi tìm kiếm từ khóa thông thường không đủ chính xác
- Nền tảng để mở rộng thành agent có khả năng gọi tool (tra cứu số liệu, gọi
  API nghiệp vụ) thay vì chỉ trả lời tĩnh
- Bộ khung đánh giá (retrieval hit-rate, citation coverage, faithfulness) để
  đo lường và cải thiện chất lượng hệ thống RAG một cách có kiểm chứng, thay
  vì đánh giá cảm tính

## Architecture

```text
Documents (JSONL)
        |
        v
Document loader -> Retriever (pluggable: lexical | embedding | hybrid) -> Context selector
                                                        |
                                                        v
                               Extractive answer generator + citations
                                                        |
                                                        v
                                      FastAPI /query and /evaluate

User query -> Tool router -> registered domain tool (when applicable)
```

The retriever is selected via `build_retriever(...)` and defaults to the deterministic
lexical (TF-IDF-style) baseline everywhere (app, Docker, CI) so the service stays fully
offline out of the box. A Vietnamese sentence-embedding retriever
(`bkai-foundation-models/vietnamese-bi-encoder`, via `sentence-transformers`) and a
hybrid retriever (weighted lexical + cosine score) are available as opt-in modes. A
future milestone will add an interchangeable LLM provider, reranking, and online
experiment tracking without changing the API contract.

### Enabling embedding / hybrid retrieval

```bash
pip install -e '.[embeddings]'   # optional, pulls in sentence-transformers/torch
VSF_RETRIEVER=hybrid uvicorn vsf_rag.api:app --app-dir src --reload
# VSF_RETRIEVER accepts: lexical (default) | embedding | hybrid
```

`GET /health` reports the active retriever class (`"retriever": "LexicalRetriever"`,
`"EmbeddingRetriever"`, or `"HybridRetriever"`) for quick verification.

Compare hit-rate/MRR@3 across all three modes on the checked-in eval set:

```bash
python scripts/compare_retrievers.py
```

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
