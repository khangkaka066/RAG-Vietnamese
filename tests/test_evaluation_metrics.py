from __future__ import annotations

import builtins
import csv
from pathlib import Path

import pytest

from vsf_rag.answering import GroundedAnswerEngine, build_answer_engine
from vsf_rag.evaluation import evaluate, load_eval_cases, write_csv_report, write_json_report
from vsf_rag.metrics import idf_weighted_containment, mrr_at_k
from vsf_rag.retrieval import LexicalRetriever, load_documents
from vsf_rag.router import build_router
from vsf_rag.tools import build_default_registry


ROOT = Path(__file__).resolve().parents[1]


def _documents():
    return load_documents(ROOT / "data" / "knowledge_base.jsonl")


def _idf():
    from vsf_rag.metrics import build_idf

    return build_idf(_documents())


# --- idf_weighted_containment ------------------------------------------------


def test_idf_weighted_containment_full_overlap_is_one() -> None:
    idf = {"a": 1.0, "b": 2.0}
    assert idf_weighted_containment("a b", "a b c", idf, oov_idf=3.0) == 1.0


def test_idf_weighted_containment_disjoint_is_zero() -> None:
    idf = {"a": 1.0, "b": 2.0}
    assert idf_weighted_containment("a b", "c d", idf, oov_idf=3.0) == 0.0


def test_idf_weighted_containment_empty_candidate_is_zero() -> None:
    idf = {"a": 1.0}
    assert idf_weighted_containment("", "a b", idf, oov_idf=3.0) == 0.0


def test_idf_weighted_containment_bounded_in_unit_interval() -> None:
    idf = _idf()
    score = idf_weighted_containment("RAG dùng context và trích dẫn", "một câu ngẫu nhiên", idf, oov_idf=5.0)
    assert 0.0 <= score <= 1.0


# --- mrr_at_k -----------------------------------------------------------------


def test_mrr_at_k_rank_one() -> None:
    rr, rank = mrr_at_k(["a", "b", "c"], {"a"}, k=3)
    assert rr == 1.0
    assert rank == 1


def test_mrr_at_k_rank_three() -> None:
    rr, rank = mrr_at_k(["a", "b", "c"], {"c"}, k=3)
    assert rr == pytest.approx(1 / 3)
    assert rank == 3


def test_mrr_at_k_no_hit() -> None:
    rr, rank = mrr_at_k(["a", "b", "c"], {"z"}, k=3)
    assert rr == 0.0
    assert rank is None


def test_mrr_at_k_respects_k() -> None:
    rr, rank = mrr_at_k(["a", "b", "c"], {"c"}, k=2)
    assert rr == 0.0
    assert rank is None


# --- evaluate() -----------------------------------------------------------


def test_evaluate_report_schema_and_ranges() -> None:
    engine = GroundedAnswerEngine(LexicalRetriever(_documents()))
    cases = load_eval_cases(ROOT / "data" / "eval.jsonl")

    report = evaluate(engine, cases, top_k=3)

    # Old keys stay present (backward compatibility).
    for key in ("cases", "retrieval_hit_rate", "citation_coverage", "answer_keyword_coverage", "details"):
        assert key in report

    # New keys are present.
    for key in (
        "top_k",
        "retrieval_mrr_at_k",
        "citation_coverage_model_asserted",
        "faithfulness",
        "faithfulness_scored_cases",
        "answer_relevancy",
        "latency_ms",
        "routes",
    ):
        assert key in report

    assert report["cases"] == len(cases)
    assert len(report["details"]) == len(cases)
    assert 0.0 <= report["faithfulness"] <= 1.0
    assert 0.0 <= report["answer_relevancy"] <= 1.0
    assert report["retrieval_mrr_at_k"] <= report["retrieval_hit_rate"]

    for key in ("retrieval_mean", "retrieval_p95", "generation_mean", "generation_p95", "total_mean", "total_p95"):
        assert report["latency_ms"][key] >= 0.0


def test_evaluate_unavailable_case_has_no_faithfulness_and_is_excluded_from_average() -> None:
    engine = GroundedAnswerEngine(LexicalRetriever(_documents()))
    cases = [{"query": "xyzabc qwerty asdf lorem ipsum", "expected_doc_ids": [], "required_terms": []}]

    report = evaluate(engine, cases, top_k=3)

    assert report["details"][0]["status"] == "UNAVAILABLE"
    assert report["details"][0]["faithfulness"] is None
    assert report["faithfulness_scored_cases"] == 0
    assert report["faithfulness"] == 0.0


def test_evaluate_tool_route_skips_retrieval_latency() -> None:
    engine = build_answer_engine(
        LexicalRetriever(_documents()),
        router=build_router(),
        tools=build_default_registry(),
    )
    cases = [{"query": "2 + 3 * 4", "expected_doc_ids": [], "required_terms": []}]

    report = evaluate(engine, cases, top_k=3)

    detail = report["details"][0]
    assert detail["route"] == "tool"
    assert detail["latency_ms"]["retrieval"] == 0.0
    assert report["routes"] == {"tool": 1}


# --- writers ----------------------------------------------------------------


def test_write_json_report_round_trips(tmp_path) -> None:
    engine = GroundedAnswerEngine(LexicalRetriever(_documents()))
    cases = load_eval_cases(ROOT / "data" / "eval.jsonl")
    report = evaluate(engine, cases, top_k=3)

    destination = tmp_path / "nested" / "eval.json"
    write_json_report(report, destination)

    import json

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["cases"] == report["cases"]


def test_write_csv_report_has_one_row_per_case(tmp_path) -> None:
    engine = GroundedAnswerEngine(LexicalRetriever(_documents()))
    cases = load_eval_cases(ROOT / "data" / "eval.jsonl")
    report = evaluate(engine, cases, top_k=3)

    destination = tmp_path / "nested" / "eval.csv"
    write_csv_report(report, destination)

    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert len(rows) == len(cases) + 1


# --- tracking -----------------------------------------------------------------


def test_log_report_is_noop_without_mlflow(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("mlflow not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from vsf_rag.tracking import log_report

    assert log_report({"retrieval_hit_rate": 1.0}) is False
