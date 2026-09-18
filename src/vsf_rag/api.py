from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .answering import build_answer_engine, GroundedAnswerEngine
from .retrieval import build_retriever, load_documents
from .tools import ToolRegistry


ROOT = Path(__file__).resolve().parents[2]
documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
tools = ToolRegistry()


@lru_cache(maxsize=1)
def get_engine() -> GroundedAnswerEngine:
    """Lazily build (and memoize) the process-lifetime answer engine.

    Deliberately *not* built at module import time: ``build_answer_engine``
    reads ``OPENROUTER_API_KEY``/``OPENROUTER_MODEL`` from the environment,
    and pytest imports this module during test *collection* -- before any
    test's fixtures (e.g. the autouse env-cleanup fixture in
    ``tests/conftest.py``) have run. Deferring construction to first use
    ensures env-based credential isolation in tests actually takes effect.
    """
    return build_answer_engine(build_retriever(documents))
tools.register(
    "list_sources",
    "List the sources currently indexed by the service.",
    lambda: {"sources": sorted({document.source for document in documents})},
)

app = FastAPI(title="Vietnamese RAG & LLM Evaluation Service", version="0.1.0")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class ToolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    arguments: dict = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "documents": len(documents),
        "version": app.version,
        "retriever": type(engine.retriever).__name__,
        "generator": "llm" if engine.llm is not None else "extractive",
    }


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return get_engine().answer(request.query, request.top_k).to_dict()


@app.get("/tools")
def list_tools() -> dict:
    return {"tools": tools.describe()}


@app.post("/tools/call")
def call_tool(request: ToolRequest) -> dict:
    return {"tool": request.name, "result": tools.call(request.name, request.arguments)}
