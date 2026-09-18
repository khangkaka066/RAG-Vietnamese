from __future__ import annotations

from pathlib import Path

import pytest

from vietnamese_rag.answering import GroundedAnswerEngine, build_answer_engine
from vietnamese_rag.llm import ContextChunk, LLMAnswer, LLMError, OpenRouterProvider
from vietnamese_rag.retrieval import LexicalRetriever, load_documents


ROOT = Path(__file__).resolve().parents[1]


def _documents():
    return load_documents(ROOT / "data" / "knowledge_base.jsonl")


class _StubLLM:
    def __init__(self, answer: LLMAnswer | None = None, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error
        self.calls: list[tuple[str, list[ContextChunk]]] = []

    def generate(self, query: str, context: list[ContextChunk]) -> LLMAnswer:
        self.calls.append((query, context))
        if self._error is not None:
            raise self._error
        return self._answer


def test_engine_uses_llm_answer_when_provider_succeeds() -> None:
    retriever = LexicalRetriever(_documents())
    stub = _StubLLM(LLMAnswer(text="Câu trả lời LLM.", citation_ids=["rag-001"], insufficient=False))
    engine = GroundedAnswerEngine(retriever, llm=stub)

    result = engine.answer("RAG cần trích dẫn nguồn như thế nào?")

    assert result.status == "OK"
    assert result.answer == "Câu trả lời LLM."
    assert result.citations == ["internal-demo/rag-principles.md"]
    assert result.generator == "llm"
    assert len(stub.calls) == 1


def test_engine_falls_back_to_top1_citation_when_all_citations_hallucinated() -> None:
    retriever = LexicalRetriever(_documents())
    stub = _StubLLM(
        LLMAnswer(text="Câu trả lời LLM.", citation_ids=["not-a-real-id"], insufficient=False)
    )
    engine = GroundedAnswerEngine(retriever, llm=stub)

    result = engine.answer("RAG cần trích dẫn nguồn như thế nào?")

    # None of the model's citation ids matched the supplied context, but the
    # model did not flag the answer as insufficient. We must still return
    # status "OK" while keeping traceability by falling back to the
    # top-ranked retrieved chunk (plan decision 5).
    assert result.status == "OK"
    assert result.citations == ["internal-demo/rag-principles.md"]
    assert result.generator == "llm"


def test_engine_marks_unavailable_when_llm_reports_insufficient() -> None:
    retriever = LexicalRetriever(_documents())
    stub = _StubLLM(LLMAnswer(text="Không đủ dữ liệu.", citation_ids=[], insufficient=True))
    engine = GroundedAnswerEngine(retriever, llm=stub)

    result = engine.answer("RAG cần trích dẫn nguồn như thế nào?")

    assert result.status == "UNAVAILABLE"
    assert result.citations == []
    assert result.confidence == 0.0
    assert result.generator == "llm"


def test_engine_falls_back_to_extractive_when_llm_raises_llm_error() -> None:
    retriever = LexicalRetriever(_documents())
    stub = _StubLLM(error=LLMError("boom"))
    engine = GroundedAnswerEngine(retriever, llm=stub)

    result = engine.answer("RAG cần trích dẫn nguồn như thế nào?")

    assert result.status == "OK"
    assert result.citations == ["internal-demo/rag-principles.md"]
    assert "Theo tài liệu" in result.answer
    assert result.generator == "extractive"


def test_engine_stays_extractive_without_llm_provider() -> None:
    retriever = LexicalRetriever(_documents())
    engine = GroundedAnswerEngine(retriever)

    result = engine.answer("RAG cần trích dẫn nguồn như thế nào?")

    assert result.status == "OK"
    assert "Theo tài liệu" in result.answer
    assert result.generator == "extractive"


def test_engine_still_unavailable_when_no_evidence_regardless_of_llm() -> None:
    retriever = LexicalRetriever(_documents())
    stub = _StubLLM()
    engine = GroundedAnswerEngine(retriever, llm=stub)

    result = engine.answer("Thời tiết hôm nay ở sao Hỏa thế nào?")

    assert result.status == "UNAVAILABLE"
    assert result.citations == []
    # The LLM must not even be called when retrieval found no usable evidence.
    assert stub.calls == []


def test_build_answer_engine_defaults_to_extractive_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    engine = build_answer_engine(LexicalRetriever(_documents()))
    assert engine.llm is None


def test_build_answer_engine_builds_openrouter_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model:free")
    engine = build_answer_engine(LexicalRetriever(_documents()))

    assert isinstance(engine.llm, OpenRouterProvider)
    assert engine.llm.api_key == "env-key"
    assert engine.llm.model == "some/model:free"
    assert str(engine.llm.client.base_url) == "https://openrouter.ai/api/v1/"


def test_build_answer_engine_explicit_llm_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    stub = _StubLLM()
    engine = build_answer_engine(LexicalRetriever(_documents()), llm=stub)
    assert engine.llm is stub
