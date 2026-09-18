from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .llm import ContextChunk, LLMError, LLMProvider, build_llm_provider
from .retrieval import Retriever, SearchResult
from .router import RouteDecision, Router, build_router
from .tools import ToolRegistry, build_default_registry


# Used as ``route_reason`` when no router is wired into the engine at all
# (bare ``GroundedAnswerEngine(retriever)`` construction) -- distinguishes
# "router evaluated and chose rag" from "there was no router to ask".
_NO_ROUTER_REASON = "Router chưa được cấu hình; mặc định RAG."


@dataclass(frozen=True)
class Answer:
    query: str
    answer: str
    citations: list[str]
    confidence: float
    retrieved_documents: list[dict]
    status: str
    generator: str = "extractive"
    # False only when the LLM returned no valid citation id and the engine
    # substituted the top-ranked retrieved chunk instead (see
    # _answer_with_llm below). Lets evaluation.py distinguish a citation the
    # model actually asserted from this low-trust fallback.
    citations_are_model_asserted: bool = True
    # Which path produced this answer: "rag" (retrieval + generation, Phase
    # 2 behavior, unchanged) or "tool" (a registered tool handled the query
    # directly, no retrieval involved). Always "rag" when no ``router`` is
    # wired into the engine (see ``GroundedAnswerEngine.__init__``).
    route: str = "rag"
    route_reason: str = ""
    # Auditable log of every tool call attempted while answering this query
    # (``ToolCallTrace.to_dict()`` entries). Empty for the "rag" route.
    tool_trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _render_tool_answer(tool_name: str, result: dict) -> str:
    """Render a successful tool result as a short Vietnamese sentence.

    Falls back to a generic ``str(result)`` rendering for any tool not
    explicitly special-cased here, so adding a new tool never breaks
    ``_answer_with_tool`` -- it just gets a less polished default sentence
    until a dedicated case is added.
    """
    if tool_name == "calculate":
        return f"Kết quả: {result['expression']} = {result['result']}."
    if tool_name == "current_datetime":
        day, month, year = result["date"].split("-")[::-1]
        return (
            f"Bây giờ là {result['time']} ngày {day}/{month}/{year} "
            f"({result['weekday']}, múi giờ {result['timezone']})."
        )
    return f"Kết quả từ tool '{tool_name}': {result}"


class GroundedAnswerEngine:
    """Answer generation grounded in retrieval, with two interchangeable backends.

    When an ``LLMProvider`` is configured, ``answer`` generates the response
    text via the LLM (still restricted to retrieved evidence and citation
    ids). Otherwise -- or if the LLM call fails -- it falls back to the
    deterministic extractive baseline. Retrieval, the ``minimum_score``
    threshold, the ``UNAVAILABLE`` status, and the API contract (``Answer``)
    stay identical in both cases.
    """

    def __init__(
        self,
        retriever: Retriever,
        minimum_score: float | None = None,
        llm: LLMProvider | None = None,
        router: Router | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.retriever = retriever
        self.minimum_score = (
            minimum_score if minimum_score is not None else getattr(retriever, "minimum_score", 0.15)
        )
        self.llm = llm
        # Both default to ``None``: an engine built with the bare
        # constructor (as every Phase 1/2 test does) keeps the exact old
        # behavior -- every query goes through "rag" -- so wiring the router
        # in is opt-in and cannot regress existing callers.
        self.router = router
        self.tools = tools

    def answer(self, query: str, top_k: int = 3) -> Answer:
        decision: RouteDecision | None = None
        if self.router is not None and self.tools is not None:
            decision = self.router.route(query)
            if decision.route == "tool":
                return self._answer_with_tool(query, decision)

        route_reason = decision.reason if decision is not None else _NO_ROUTER_REASON
        return self._answer_with_rag(query, top_k, route_reason)

    def _answer_with_tool(self, query: str, decision: RouteDecision) -> Answer:
        trace = self.tools.call_with_trace(decision.tool_name, decision.arguments)

        if trace.error is not None:
            return Answer(
                query=query,
                answer=f"Không thể thực hiện tool '{decision.tool_name}': {trace.error}",
                citations=[],
                confidence=0.0,
                retrieved_documents=[],
                status="UNAVAILABLE",
                route="tool",
                route_reason=decision.reason,
                tool_trace=[trace.to_dict()],
            )

        return Answer(
            query=query,
            answer=_render_tool_answer(decision.tool_name, trace.result),
            citations=[],
            confidence=1.0,
            retrieved_documents=[],
            status="OK",
            generator="tool",
            route="tool",
            route_reason=decision.reason,
            tool_trace=[trace.to_dict()],
        )

    def _answer_with_rag(self, query: str, top_k: int, route_reason: str) -> Answer:
        results = self.retriever.search(query, top_k=top_k)
        usable = [result for result in results if result.score >= self.minimum_score]
        retrieved = [
            {
                "id": result.document.id,
                "title": result.document.title,
                "source": result.document.source,
                "score": round(result.score, 4),
                "matched_terms": list(result.matched_terms),
            }
            for result in results
        ]
        if not usable:
            return Answer(
                query=query,
                answer="Tôi chưa tìm thấy đủ bằng chứng trong cơ sở tri thức để trả lời câu hỏi này.",
                citations=[],
                confidence=0.0,
                retrieved_documents=retrieved,
                status="UNAVAILABLE",
                route_reason=route_reason,
            )

        best = usable[0]
        confidence = round(min(1.0, best.score / 1.5), 4)

        if self.llm is not None:
            try:
                return self._answer_with_llm(query, usable, retrieved, confidence, route_reason)
            except LLMError:
                pass

        return self._answer_extractive(query, best, retrieved, confidence, route_reason)

    def _answer_extractive(
        self,
        query: str,
        best: SearchResult,
        retrieved: list[dict],
        confidence: float,
        route_reason: str,
    ) -> Answer:
        return Answer(
            query=query,
            answer=f"Theo tài liệu '{best.document.title}': {best.document.text}",
            citations=[best.document.source],
            confidence=confidence,
            retrieved_documents=retrieved,
            status="OK",
            route_reason=route_reason,
        )

    def _answer_with_llm(
        self,
        query: str,
        usable: list[SearchResult],
        retrieved: list[dict],
        confidence: float,
        route_reason: str,
    ) -> Answer:
        context = [
            ContextChunk(
                citation_id=result.document.id,
                title=result.document.title,
                source=result.document.source,
                text=result.document.text,
            )
            for result in usable
        ]
        source_by_id = {result.document.id: result.document.source for result in usable}

        llm_answer = self.llm.generate(query, context)

        if llm_answer.insufficient:
            return Answer(
                query=query,
                answer=llm_answer.text,
                citations=[],
                confidence=0.0,
                retrieved_documents=retrieved,
                status="UNAVAILABLE",
                generator="llm",
                route_reason=route_reason,
            )

        citations = [
            source_by_id[citation_id]
            for citation_id in llm_answer.citation_ids
            if citation_id in source_by_id
        ]

        model_asserted = True
        if not citations:
            # The model answered (it did not flag "insufficient"), but every
            # citation id it returned was hallucinated / not part of the
            # supplied context. Rather than dropping traceability entirely
            # (which would also silently tank the citation_coverage metric,
            # see evaluation.py:37), fall back to the top-ranked retrieved
            # chunk as a low-trust-but-safe citation. Status stays "OK" per
            # plan decision 5, but the fallback is flagged as not
            # model-asserted so callers can tell it apart from a real
            # citation.
            citations = [usable[0].document.source]
            model_asserted = False

        return Answer(
            query=query,
            answer=llm_answer.text,
            citations=citations,
            confidence=confidence,
            retrieved_documents=retrieved,
            status="OK",
            generator="llm",
            citations_are_model_asserted=model_asserted,
            route_reason=route_reason,
        )


def build_answer_engine(
    retriever: Retriever,
    minimum_score: float | None = None,
    llm: LLMProvider | None = None,
    router: Router | None = None,
    tools: ToolRegistry | None = None,
    enable_router: bool = True,
) -> GroundedAnswerEngine:
    """Factory that wires up the LLM provider from the environment by default.

    Explicitly passing ``llm=None`` still triggers the environment-based
    auto-detection (``build_llm_provider()``); to force the extractive-only
    path regardless of environment variables, construct
    ``GroundedAnswerEngine`` directly instead.

    ``router``/``tools`` default to the built-in rule router and tool
    registry (``build_router()``/``build_default_registry()``) so the
    tool-use path is enabled out of the box -- mirroring the same
    "explicit None still gets a default" convention as ``llm`` above. To
    avoid repeating the ``llm=None`` ambiguity trap (no way to tell "didn't
    pass it" from "deliberately disabled"), disabling the router/tool-use
    path entirely is a *separate*, explicit ``enable_router=False`` flag
    rather than overloading ``router=None``/``tools=None``.
    """
    resolved_llm = llm if llm is not None else build_llm_provider()
    if enable_router:
        resolved_router = router if router is not None else build_router()
        resolved_tools = tools if tools is not None else build_default_registry()
    else:
        resolved_router = None
        resolved_tools = None
    return GroundedAnswerEngine(
        retriever,
        minimum_score=minimum_score,
        llm=resolved_llm,
        router=resolved_router,
        tools=resolved_tools,
    )
