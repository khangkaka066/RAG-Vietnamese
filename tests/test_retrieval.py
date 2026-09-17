from pathlib import Path

from vsf_rag.answering import GroundedAnswerEngine
from vsf_rag.evaluation import evaluate, load_eval_cases
from vsf_rag.retrieval import LexicalRetriever, load_documents, tokenize


ROOT = Path(__file__).resolve().parents[1]


def build_engine() -> GroundedAnswerEngine:
    documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
    return GroundedAnswerEngine(LexicalRetriever(documents))


def test_tokenizer_preserves_vietnamese_terms() -> None:
    assert "đánh" in tokenize("Đánh giá hệ thống")
    assert "llm" in tokenize("LLM")


def test_retrieval_returns_expected_document() -> None:
    result = build_engine().answer("RAG cần trích dẫn nguồn như thế nào?")
    assert result.status == "OK"
    assert result.retrieved_documents[0]["id"] == "rag-001"
    assert result.citations == ["internal-demo/rag-principles.md"]


def test_unavailable_when_no_evidence_exists() -> None:
    result = build_engine().answer("Thời tiết hôm nay ở sao Hỏa thế nào?")
    assert result.status == "UNAVAILABLE"
    assert result.citations == []


def test_evaluation_report_has_quality_metrics() -> None:
    eval_cases = load_eval_cases(ROOT / "data" / "eval.jsonl")
    report = evaluate(build_engine(), eval_cases)
    assert report["cases"] == len(eval_cases)
    assert report["retrieval_hit_rate"] >= 0.8
    assert report["citation_coverage"] == 1.0
