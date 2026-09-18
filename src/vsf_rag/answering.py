from __future__ import annotations

from dataclasses import asdict, dataclass

from .llm import ContextChunk, LLMError, LLMProvider, build_llm_provider
from .retrieval import Retriever, SearchResult


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

    def to_dict(self) -> dict:
        return asdict(self)


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
    ) -> None:
        self.retriever = retriever
        self.minimum_score = (
            minimum_score if minimum_score is not None else getattr(retriever, "minimum_score", 0.15)
        )
        self.llm = llm

    def answer(self, query: str, top_k: int = 3) -> Answer:
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
            )

        best = usable[0]
        confidence = round(min(1.0, best.score / 1.5), 4)

        if self.llm is not None:
            try:
                return self._answer_with_llm(query, usable, retrieved, confidence)
            except LLMError:
                pass

        return self._answer_extractive(query, best, retrieved, confidence)

    def _answer_extractive(
        self,
        query: str,
        best: SearchResult,
        retrieved: list[dict],
        confidence: float,
    ) -> Answer:
        return Answer(
            query=query,
            answer=f"Theo tài liệu '{best.document.title}': {best.document.text}",
            citations=[best.document.source],
            confidence=confidence,
            retrieved_documents=retrieved,
            status="OK",
        )

    def _answer_with_llm(
        self,
        query: str,
        usable: list[SearchResult],
        retrieved: list[dict],
        confidence: float,
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
        )


def build_answer_engine(
    retriever: Retriever,
    minimum_score: float | None = None,
    llm: LLMProvider | None = None,
) -> GroundedAnswerEngine:
    """Factory that wires up the LLM provider from the environment by default.

    Explicitly passing ``llm=None`` still triggers the environment-based
    auto-detection (``build_llm_provider()``); to force the extractive-only
    path regardless of environment variables, construct
    ``GroundedAnswerEngine`` directly instead.
    """
    resolved_llm = llm if llm is not None else build_llm_provider()
    return GroundedAnswerEngine(retriever, minimum_score=minimum_score, llm=resolved_llm)
