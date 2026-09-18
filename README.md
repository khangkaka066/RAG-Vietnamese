# Vietnamese RAG & LLM Evaluation Service

Status: **In Progress**

A small, provider-agnostic service for Vietnamese document retrieval, grounded answers, citations, tool calls, and repeatable evaluation.

## Real-world applications

Once complete, this architecture can be applied to problems that require
retrieving and answering from an internal Vietnamese document store, such as:

- An internal policy/procedure lookup assistant for enterprises (HR, legal, operations)
- A customer support chatbot grounded in product documentation/FAQs, with source
  citations to verify answers instead of hallucinating
- A semantic search tool for technical documents, contracts, or internal
  knowledge bases where plain keyword search isn't accurate enough
- A foundation for extending into a tool-using agent (looking up data, calling
  business APIs) instead of only returning static answers
- An evaluation framework (retrieval hit-rate, citation coverage, faithfulness)
  to measure and improve RAG system quality with evidence, rather than by
  subjective judgment

## Architecture

```text
                                   User query
                                       |
                                       v
                     RuleRouter (rag vs tool, rule-based, offline)
                            |                       |
                    route == "rag"          route == "tool"
                            |                       |
                            v                       v
       Document loader -> Retriever          Registered tool (calculate |
       (lexical | embedding | hybrid)        current_datetime | ...) via
                 |                            ToolRegistry.call_with_trace
                 v                                    |
       LLM generator (OpenRouter) with citations,      |
       falling back to the extractive generator        |
       when no OPENROUTER_API_KEY is set or the         |
       LLM call fails                                   |
                 |                                       |
                 +-------------------+--------------------+
                                     v
                     Answer (route, route_reason, tool_trace)
                                     |
                                     v
                          FastAPI /query, /tools, /tools/call, /evaluate
```

The retriever is selected via `build_retriever(...)` and defaults to the deterministic
lexical (TF-IDF-style) baseline everywhere (app, Docker, CI) so the service stays fully
offline out of the box. A Vietnamese sentence-embedding retriever
(`bkai-foundation-models/vietnamese-bi-encoder`, via `sentence-transformers`) and a
hybrid retriever (weighted lexical + cosine score) are available as opt-in modes.

Answer generation is selected via `build_answer_engine(...)` and defaults to the
same offline, extractive baseline unless `OPENROUTER_API_KEY` is set, in which case
it calls an OpenRouter-hosted LLM (OpenAI-compatible `/chat/completions`) with a
Vietnamese system prompt that restricts the model to the retrieved context and
requires citation ids. Any LLM/network/parsing failure (`LLMError`) transparently
falls back to the extractive generator, so the API never hard-fails because of the
LLM provider. A future milestone will add reranking and online experiment tracking
without changing the API contract.

### Enabling LLM-based generation (OpenRouter)

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)
export $(grep -v '^#' .env | xargs)
uvicorn vsf_rag.api:app --app-dir src --reload
```

`OPENROUTER_MODEL` is optional and defaults to `openrouter/free` (OpenRouter's
official free auto-router). Without `OPENROUTER_API_KEY`, the service runs fully
offline using the rule-based extractive generator -- no code change required.

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

## Agent / tool routing

Every `/query` call first passes through a deterministic, offline **rule-based router**
(`RuleRouter` in `src/vsf_rag/router.py`) that classifies the query as either `"rag"`
(the Phase 1/2 retrieval+generation pipeline, unchanged) or `"tool"` (a registered tool
handles the query directly, with no retrieval call at all). Two tools are registered by
default (`src/vsf_rag/tools.py`):

- `calculate(expression)` — evaluates a whitelisted arithmetic expression
  (`+ - * / // % **`, parentheses) via a restricted `ast` walker. No `eval()`/`exec()`
  is ever used, so names, calls, attributes, and imports are always rejected.
- `current_datetime(timezone)` — looks up the current date/time in an IANA timezone
  (stdlib `zoneinfo`, defaults to `Asia/Ho_Chi_Minh`).

Every tool invocation (successful or not) is captured as a `ToolCallTrace` and returned
in the response, so the decision process stays fully auditable:

```json
{
  "query": "tính 2+3*4",
  "answer": "Kết quả: 2+3*4 = 14.",
  "status": "OK",
  "generator": "tool",
  "route": "tool",
  "route_reason": "Câu hỏi là một biểu thức số học; dùng tool calculate.",
  "tool_trace": [
    {"tool": "calculate", "arguments": {"expression": "2+3*4"}, "result": {"expression": "2+3*4", "result": 14}, "error": null, "duration_ms": 0.04}
  ]
}
```

A tool failure (e.g. `"tính 1/0"`) never crashes the API: it returns `status:
"UNAVAILABLE"` with the failure captured in `tool_trace[0]["error"]`. `GET /tools` and
`POST /tools/call` (the router and these endpoints share one `ToolRegistry`) let you
list/call tools directly; `POST /tools/call` returns `404` for an unknown tool name and
`400` for a `ToolError`/bad-argument failure instead of a `500`.

**Limitations**: the router is a small set of hand-written rules (keyword/regex
matching), not an LLM performing function-calling/tool-selection -- it is deliberately
simple and fully offline, and can be wrong on phrasing it wasn't written for. It does
not chain multiple tool calls or mix a tool result with retrieved context in one
answer. Constructing `GroundedAnswerEngine` directly (bypassing `build_answer_engine`)
leaves `router`/`tools` unset and keeps the old "always rag" behavior; passing
`enable_router=False` to `build_answer_engine` disables the tool-use path explicitly.

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

The report includes retrieval hit-rate@k, MRR@k, citation coverage, per-stage
latency (retrieval / generation / total, mean + p95), and two RAGAS-style
generation metrics computed with an **offline, IDF-weighted lexical
heuristic** (not an LLM-as-judge):

- `faithfulness` — how much of the answer's content is grounded in the
  retrieved context actually shown to the generator.
- `answer_relevancy` — how much of the query's content the answer addresses.

To also write the report to disk and log it to MLflow (`pip install -e
'.[tracking]'` first):

```bash
python scripts/run_eval.py --top-k 3 \
  --json-out reports/eval.json --csv-out reports/eval.csv --mlflow
```

## Docker

```bash
docker build -t vietnamese-rag-eval .
docker run --rm -p 8000:8000 vietnamese-rag-eval
```

## Current quality gates

- Retrieval hit-rate@k and MRR@k on the checked-in evaluation set
- Answer keyword coverage
- Citation coverage
- Faithfulness and answer relevancy (offline lexical heuristic)
- Per-stage latency (retrieval / generation / total)
- Unavailable behavior when evidence is insufficient
- Unit tests for tokenization, ranking, evaluation, and failure handling

## Roadmap

1. Add multilingual embedding retrieval and a cross-encoder reranker.
2. ~~Add an LLM provider interface with local-model and API-backed implementations.~~
   Done: OpenRouter-backed `LLMProvider`, with automatic extractive fallback.
3. ~~Add Vietnamese RAG faithfulness and answer-relevance evaluation.~~
   Done: offline IDF-weighted lexical heuristic (see "Run the evaluation set" above).
4. ~~Add MLflow/W&B experiment tracking and latency metrics.~~
   Done: `--mlflow` flag on `scripts/run_eval.py`, retrieval/generation/total latency in the report.
5. ~~Add a tool-using agent with explicit planning, tool-call validation, and trace logging.~~
   Done: rule-based router + `calculate`/`current_datetime` tools with full call-trace logging
   (see "Agent / tool routing" above).
6. Add GitHub Actions, OpenAPI examples, and a small deployed demo.
