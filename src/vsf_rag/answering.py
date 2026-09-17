from __future__ import annotations

from dataclasses import asdict, dataclass

from .retrieval import LexicalRetriever, SearchResult


@dataclass(frozen=True)
class Answer:
    query: str
    answer: str
    citations: list[str]
    confidence: float
    retrieved_documents: list[dict]
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


class GroundedAnswerEngine:
    """Extractive baseline with an explicit unavailable state.

    An LLM provider can replace ``_compose`` later while keeping retrieval,
    citations, status handling, and the API contract stable.
    """

    def __init__(self, retriever: LexicalRetriever, minimum_score: float = 0.15) -> None:
        self.retriever = retriever
        self.minimum_score = minimum_score

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
        confidence = min(1.0, best.score / 1.5)
        return Answer(
            query=query,
            answer=f"Theo tài liệu '{best.document.title}': {best.document.text}",
            citations=[best.document.source],
            confidence=round(confidence, 4),
            retrieved_documents=retrieved,
            status="OK",
        )
