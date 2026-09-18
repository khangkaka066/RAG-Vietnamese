from __future__ import annotations

from pathlib import Path

from vsf_rag.answering import GroundedAnswerEngine, build_answer_engine
from vsf_rag.retrieval import LexicalRetriever, load_documents
from vsf_rag.router import RouteDecision, Router
from vsf_rag.tools import ToolRegistry, build_default_registry


ROOT = Path(__file__).resolve().parents[1]


def _documents():
    return load_documents(ROOT / "data" / "knowledge_base.jsonl")


class _SpyRetriever(LexicalRetriever):
    """Wraps ``LexicalRetriever`` to count how many times ``search`` runs."""

    def __init__(self, documents) -> None:
        super().__init__(documents)
        self.search_calls = 0

    def search(self, query: str, top_k: int = 3):
        self.search_calls += 1
        return super().search(query, top_k=top_k)


class _StaticRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self._decision = decision

    def route(self, query: str) -> RouteDecision:
        return self._decision


def test_arithmetic_query_routes_to_tool_and_skips_retrieval() -> None:
    retriever = _SpyRetriever(_documents())
    engine = GroundedAnswerEngine(retriever, router=_StaticRouter(
        RouteDecision(route="tool", reason="test", tool_name="calculate", arguments={"expression": "2+3*4"})
    ), tools=build_default_registry())

    result = engine.answer("tính 2+3*4")

    assert result.route == "tool"
    assert result.generator == "tool"
    assert result.status == "OK"
    assert "14" in result.answer
    assert result.citations == []
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["error"] is None
    assert retriever.search_calls == 0


def test_rag_query_behavior_unchanged_from_phase_2() -> None:
    engine = build_answer_engine(LexicalRetriever(_documents()))

    result = engine.answer("RAG cần trích dẫn nguồn như thế nào?")

    assert result.generator in {"extractive", "llm"}
    assert result.route == "rag"
    assert result.route_reason
    assert result.tool_trace == []


def test_tool_error_yields_unavailable_status_with_trace_error() -> None:
    retriever = LexicalRetriever(_documents())
    engine = GroundedAnswerEngine(
        retriever,
        router=_StaticRouter(
            RouteDecision(route="tool", reason="test", tool_name="calculate", arguments={"expression": "1/0"})
        ),
        tools=build_default_registry(),
    )

    result = engine.answer("tính 1/0")

    assert result.status == "UNAVAILABLE"
    assert result.tool_trace
    assert result.tool_trace[0]["error"]


def test_engine_without_router_always_uses_rag_even_for_arithmetic_query() -> None:
    """Regression: constructing the engine the old (Phase 1/2) way must not
    silently start routing to tools once router.py/tools.py exist."""
    retriever = LexicalRetriever(_documents())
    engine = GroundedAnswerEngine(retriever)

    result = engine.answer("2+3*4")

    assert result.route == "rag"
    assert result.tool_trace == []


def test_build_answer_engine_defaults_to_router_and_tools_wired() -> None:
    engine = build_answer_engine(LexicalRetriever(_documents()))
    assert engine.router is not None
    assert engine.tools is not None
    assert isinstance(engine.tools, ToolRegistry)


def test_build_answer_engine_enable_router_false_disables_tool_routing() -> None:
    engine = build_answer_engine(LexicalRetriever(_documents()), enable_router=False)
    assert engine.router is None
    assert engine.tools is None

    result = engine.answer("2+3*4")
    assert result.route == "rag"
