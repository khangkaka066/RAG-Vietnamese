from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .answering import GroundedAnswerEngine
from .metrics import answer_relevancy, build_idf, default_idf, faithfulness, mrr_at_k, percentile
from .retrieval import Retriever, SearchResult


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class _TimedRetriever:
    """Wraps a ``Retriever`` to measure time actually spent inside ``search``.

    Used to attribute latency correctly when the router sends a query down
    the "tool" path: ``search`` is never called in that case, so
    ``elapsed_ms`` stays ``0.0`` -- calling ``search`` separately just to
    measure it would run retrieval the real pipeline never performs.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self.documents = retriever.documents
        self.minimum_score = retriever.minimum_score
        self._elapsed_s = 0.0

    def reset(self) -> None:
        self._elapsed_s = 0.0

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_s * 1000.0

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        start = time.perf_counter()
        try:
            return self._retriever.search(query, top_k=top_k)
        finally:
            self._elapsed_s += time.perf_counter() - start


def evaluate(engine: GroundedAnswerEngine, cases: list[dict[str, Any]], top_k: int = 3) -> dict[str, Any]:
    if not cases:
        raise ValueError("Evaluation set is empty")

    documents = engine.retriever.documents
    idf = build_idf(documents)
    oov_idf = default_idf(documents)
    document_by_id = {document.id: document for document in documents}

    timed_retriever = _TimedRetriever(engine.retriever)
    # "Shadow" engine: identical configuration to ``engine``, but retrieval is
    # routed through ``timed_retriever`` so per-case latency can be measured
    # without touching ``answering.py``/``Answer``.
    shadow = GroundedAnswerEngine(
        timed_retriever,
        minimum_score=engine.minimum_score,
        llm=engine.llm,
        router=engine.router,
        tools=engine.tools,
    )

    hits = 0
    citation_hits = 0
    citation_hits_model_asserted = 0
    keyword_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    total_latencies: list[float] = []
    routes: dict[str, int] = {}
    details: list[dict[str, Any]] = []

    for case in cases:
        timed_retriever.reset()
        start = time.perf_counter()
        result = shadow.answer(case["query"], top_k)
        total_ms = (time.perf_counter() - start) * 1000.0
        retrieval_ms = timed_retriever.elapsed_ms
        generation_ms = max(total_ms - retrieval_ms, 0.0)

        retrieved_ids = [item["id"] for item in result.retrieved_documents]
        expected_ids = set(case.get("expected_doc_ids", []))
        hit = bool(set(retrieved_ids) & expected_ids)
        reciprocal_rank, rank = mrr_at_k(retrieved_ids, expected_ids, top_k)

        required_terms = [term.lower() for term in case.get("required_terms", [])]
        answer_lower = result.answer.lower()
        keyword_score = (
            sum(term in answer_lower for term in required_terms) / len(required_terms)
            if required_terms else 1.0
        )

        # Mirror the "usable" filter in ``answering.py:_answer_with_rag``: only
        # chunks that cleared ``minimum_score`` were actually shown to the
        # generator, so only those count as "context" for faithfulness.
        usable_ids = [
            item["id"] for item in result.retrieved_documents if item["score"] >= shadow.minimum_score
        ]
        case_faithfulness: float | None = None
        if result.status == "OK" and usable_ids:
            context_text = " ".join(document_by_id[doc_id].text for doc_id in usable_ids)
            case_faithfulness = round(faithfulness(result.answer, context_text, idf, oov_idf), 4)
            faithfulness_scores.append(case_faithfulness)
        case_relevancy = round(answer_relevancy(case["query"], result.answer, idf, oov_idf), 4)
        relevancy_scores.append(case_relevancy)

        hits += hit
        citation_hits += bool(result.citations)
        citation_hits_model_asserted += bool(result.citations) and result.citations_are_model_asserted
        keyword_scores.append(keyword_score)
        reciprocal_ranks.append(reciprocal_rank)
        retrieval_latencies.append(retrieval_ms)
        generation_latencies.append(generation_ms)
        total_latencies.append(total_ms)
        routes[result.route] = routes.get(result.route, 0) + 1

        details.append({
            "query": case["query"],
            "retrieval_hit": hit,
            "rank": rank,
            "reciprocal_rank": round(reciprocal_rank, 4),
            "citation_present": bool(result.citations),
            "keyword_coverage": round(keyword_score, 4),
            "faithfulness": case_faithfulness,
            "answer_relevancy": case_relevancy,
            "status": result.status,
            "route": result.route,
            "generator": result.generator,
            "latency_ms": {
                "retrieval": round(retrieval_ms, 4),
                "generation": round(generation_ms, 4),
                "total": round(total_ms, 4),
            },
        })

    count = len(cases)
    return {
        "cases": count,
        "top_k": top_k,
        "retrieval_hit_rate": round(hits / count, 4),
        "retrieval_mrr_at_k": round(sum(reciprocal_ranks) / count, 4),
        "citation_coverage": round(citation_hits / count, 4),
        "citation_coverage_model_asserted": round(citation_hits_model_asserted / count, 4),
        "answer_keyword_coverage": round(sum(keyword_scores) / count, 4),
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
        if faithfulness_scores else 0.0,
        "faithfulness_scored_cases": len(faithfulness_scores),
        "answer_relevancy": round(sum(relevancy_scores) / count, 4),
        "latency_ms": {
            "retrieval_mean": round(sum(retrieval_latencies) / count, 4),
            "retrieval_p95": round(percentile(retrieval_latencies, 95), 4),
            "generation_mean": round(sum(generation_latencies) / count, 4),
            "generation_p95": round(percentile(generation_latencies, 95), 4),
            "total_mean": round(sum(total_latencies) / count, 4),
            "total_p95": round(percentile(total_latencies, 95), 4),
        },
        "routes": routes,
        "details": details,
    }


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


_CSV_FIELDS = [
    "query",
    "status",
    "route",
    "generator",
    "retrieval_hit",
    "rank",
    "reciprocal_rank",
    "citation_present",
    "keyword_coverage",
    "faithfulness",
    "answer_relevancy",
    "retrieval_ms",
    "generation_ms",
    "total_ms",
]


def write_csv_report(report: dict[str, Any], path: str | Path) -> None:
    """Write one CSV row per case in ``report["details"]``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for detail in report["details"]:
            latency = detail["latency_ms"]
            writer.writerow({
                "query": detail["query"],
                "status": detail["status"],
                "route": detail["route"],
                "generator": detail["generator"],
                "retrieval_hit": detail["retrieval_hit"],
                "rank": detail["rank"],
                "reciprocal_rank": detail["reciprocal_rank"],
                "citation_present": detail["citation_present"],
                "keyword_coverage": detail["keyword_coverage"],
                "faithfulness": detail["faithfulness"],
                "answer_relevancy": detail["answer_relevancy"],
                "retrieval_ms": latency["retrieval"],
                "generation_ms": latency["generation"],
                "total_ms": latency["total"],
            })
