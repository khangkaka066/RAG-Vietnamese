from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answering import build_answer_engine, GroundedAnswerEngine
from .evaluation import default_eval_path, evaluate, load_eval_cases
from .retrieval import build_retriever, load_documents
from .router import build_router
from .tools import build_default_registry, ToolError


ROOT = Path(__file__).resolve().parents[2]
documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
tools = build_default_registry()
tools.register(
    "list_sources",
    "List the sources currently indexed by the service.",
    lambda: {"sources": sorted({document.source for document in documents})},
)


@lru_cache(maxsize=1)
def get_engine() -> GroundedAnswerEngine:
    """Lazily build (and memoize) the process-lifetime answer engine.

    Deliberately *not* built at module import time: ``build_answer_engine``
    reads ``OPENROUTER_API_KEY``/``OPENROUTER_MODEL`` from the environment,
    and pytest imports this module during test *collection* -- before any
    test's fixtures (e.g. the autouse env-cleanup fixture in
    ``tests/conftest.py``) have run. Deferring construction to first use
    ensures env-based credential isolation in tests actually takes effect.

    Shares the module-level ``tools`` registry (including ``list_sources``,
    registered above) with the router-driven tool-use path, so a query
    routed to a tool can reach every tool exposed via ``/tools``/``/tools/call``.
    """
    return build_answer_engine(build_retriever(documents), router=build_router(), tools=tools)


@lru_cache(maxsize=1)
def get_eval_cases() -> tuple[dict, ...]:
    """Lazily load (and memoize) the checked-in evaluation set (``data/eval.jsonl``).

    Used as the default ``/evaluate`` payload when the caller does not supply
    its own ``cases``. Cached as an immutable tuple since ``lru_cache``
    requires the return value to stay untouched between calls.
    """
    return tuple(load_eval_cases(default_eval_path()))


app = FastAPI(
    title="Vietnamese RAG & LLM Evaluation Service",
    version="0.1.0",
    description=(
        "Provider-agnostic Vietnamese document retrieval, grounded answers "
        "with citations, tool-use routing, and a repeatable offline "
        "evaluation harness (hit-rate, MRR@k, citation coverage, "
        "faithfulness, answer relevancy, per-stage latency)."
    ),
    openapi_tags=[
        {"name": "rag", "description": "Retrieval + grounded answer generation, and tool-use routing."},
        {"name": "tools", "description": "Direct access to the tool registry used by the router."},
        {"name": "evaluation", "description": "Offline evaluation harness over the checked-in (or custom) eval set."},
        {"name": "ops", "description": "Operational endpoints (health/readiness)."},
    ],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "RAG cần trích dẫn nguồn như thế nào?", "top_k": 3},
                {"query": "tính 2+3*4", "top_k": 3},
            ]
        }
    }


class ToolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    arguments: dict = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {"name": "calculate", "arguments": {"expression": "2+3*4"}}
        }
    }


class EvalCase(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    expected_doc_ids: list[str] = Field(default_factory=list)
    required_terms: list[str] = Field(default_factory=list)


class EvaluateRequest(BaseModel):
    top_k: int = Field(default=3, ge=1, le=10)
    include_details: bool = False
    # ``None`` (default) scores the checked-in ``data/eval.jsonl`` set;
    # an explicit-but-empty list is rejected (``min_length=1``) rather than
    # silently falling back to the default set. ``max_length=50`` caps
    # ad-hoc requests: with ``OPENROUTER_API_KEY`` set, each case triggers
    # one LLM call, so an unbounded list is a cost/DoS risk.
    cases: list[EvalCase] | None = Field(default=None, min_length=1, max_length=50)

    model_config = {
        "json_schema_extra": {"example": {"top_k": 3, "include_details": False}}
    }


class QueryResponse(BaseModel):
    """Mirrors ``Answer.to_dict()`` field-for-field -- see ``answering.Answer``."""

    query: str
    answer: str
    citations: list[str]
    confidence: float
    retrieved_documents: list[dict]
    status: str
    generator: str = "extractive"
    citations_are_model_asserted: bool = True
    route: str = "rag"
    route_reason: str = ""
    tool_trace: list[dict] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "tính 2+3*4",
                "answer": "Kết quả: 2+3*4 = 14.",
                "citations": [],
                "confidence": 1.0,
                "retrieved_documents": [],
                "status": "OK",
                "generator": "tool",
                "citations_are_model_asserted": True,
                "route": "tool",
                "route_reason": "Câu hỏi là một biểu thức số học; dùng tool calculate.",
                "tool_trace": [
                    {
                        "tool": "calculate",
                        "arguments": {"expression": "2+3*4"},
                        "result": {"expression": "2+3*4", "result": 14},
                        "error": None,
                        "duration_ms": 0.04,
                    }
                ],
            }
        }
    }


class HealthResponse(BaseModel):
    status: str
    documents: int
    version: str
    retriever: str
    generator: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "documents": 20,
                "version": "0.1.0",
                "retriever": "LexicalRetriever",
                "generator": "extractive",
            }
        }
    }


class ToolsResponse(BaseModel):
    tools: list[dict]

    model_config = {
        "json_schema_extra": {
            "example": {
                "tools": [
                    {"name": "calculate", "description": "Evaluate a whitelisted arithmetic expression."},
                    {"name": "current_datetime", "description": "Look up the current date/time in an IANA timezone."},
                ]
            }
        }
    }


class ToolCallResponse(BaseModel):
    tool: str
    result: dict

    model_config = {
        "json_schema_extra": {
            "example": {"tool": "calculate", "result": {"expression": "2+3*4", "result": 14}}
        }
    }


class LatencyStats(BaseModel):
    """Mirrors the ``latency_ms`` block produced by ``evaluation.evaluate()``."""

    retrieval_mean: float
    retrieval_p95: float
    generation_mean: float
    generation_p95: float
    total_mean: float
    total_p95: float


class EvaluateResponse(BaseModel):
    """Mirrors the report dict returned by ``vsf_rag.evaluation.evaluate()``.

    ``details`` is only present when the request set ``include_details=true``
    (see ``evaluate_endpoint``); it is omitted from the response entirely
    otherwise, which is why the field is optional here.
    """

    cases: int
    top_k: int
    retrieval_hit_rate: float
    retrieval_mrr_at_k: float
    citation_coverage: float
    citation_coverage_model_asserted: float
    answer_keyword_coverage: float
    faithfulness: float
    faithfulness_scored_cases: int
    answer_relevancy: float
    latency_ms: LatencyStats
    routes: dict[str, int]
    details: list[dict] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "cases": 20,
                "top_k": 3,
                "retrieval_hit_rate": 1.0,
                "retrieval_mrr_at_k": 1.0,
                "citation_coverage": 1.0,
                "citation_coverage_model_asserted": 1.0,
                "answer_keyword_coverage": 1.0,
                "faithfulness": 0.94,
                "faithfulness_scored_cases": 20,
                "answer_relevancy": 0.87,
                "latency_ms": {
                    "retrieval_mean": 0.42,
                    "retrieval_p95": 0.61,
                    "generation_mean": 1.15,
                    "generation_p95": 2.03,
                    "total_mean": 1.57,
                    "total_p95": 2.64,
                },
                "routes": {"rag": 18, "tool": 2},
            }
        }
    }


@app.get("/health", tags=["ops"], response_model=HealthResponse)
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "documents": len(documents),
        "version": app.version,
        "retriever": type(engine.retriever).__name__,
        "generator": "llm" if engine.llm is not None else "extractive",
    }


@app.post("/query", tags=["rag"], response_model=QueryResponse)
def query(request: QueryRequest) -> dict:
    """Answer a query, either grounded in retrieval ("rag") or via a tool.

    The router (see ``vsf_rag.router``) inspects the query and picks one of
    two mutually-exclusive paths -- see the ``route``/``route_reason``/
    ``tool_trace`` fields on the response for which one was taken and why.
    """
    return get_engine().answer(request.query, request.top_k).to_dict()


@app.get("/tools", tags=["tools"], response_model=ToolsResponse)
def list_tools() -> dict:
    return {"tools": tools.describe()}


@app.post("/tools/call", tags=["tools"], response_model=ToolCallResponse)
def call_tool(request: ToolRequest) -> dict:
    try:
        result = tools.call(request.name, request.arguments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ToolError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tool": request.name, "result": result}


@app.post(
    "/evaluate",
    tags=["evaluation"],
    response_model=EvaluateResponse,
    response_model_exclude_none=True,
)
def evaluate_endpoint(request: EvaluateRequest) -> dict[str, Any]:
    """Run the offline evaluation harness (see ``vsf_rag.evaluation.evaluate``).

    Defaults to scoring the checked-in evaluation set (``data/eval.jsonl``,
    20 cases); pass ``cases`` to score a custom set instead (max 50, to bound
    cost/latency -- see below). The per-case ``details`` breakdown is
    included only when ``include_details=true``, to keep the default
    response small.

    **Important**: this endpoint runs against the *same engine instance*
    serving ``/query`` (``get_engine()``), not an isolated test double. If
    ``OPENROUTER_API_KEY`` is configured, every case triggers one real LLM
    call -- so the default 20-case run makes 20 LLM calls (and a custom
    ``cases`` list up to 50 could make 50), which is slower, non-free, and
    the resulting ``faithfulness``/``answer_relevancy`` scores become
    non-deterministic across runs. Without an API key, generation falls
    back to the deterministic extractive path and the run is fast/free.
    """
    cases = (
        [case.model_dump() for case in request.cases]
        if request.cases is not None
        else list(get_eval_cases())
    )
    try:
        report = evaluate(get_engine(), cases, top_k=request.top_k)
    except ValueError as exc:
        # ``evaluate()`` raises this only when ``cases`` ends up empty (a
        # runtime/data condition, not a request-shape validation failure --
        # that's handled separately by ``EvalCase``/``min_length``/
        # ``max_length`` above, which FastAPI already turns into 422).
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report = dict(report)
    if not request.include_details:
        report.pop("details", None)
    return report
