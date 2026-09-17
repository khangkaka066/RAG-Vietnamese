from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .answering import GroundedAnswerEngine


def load_eval_cases(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(engine: GroundedAnswerEngine, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("Evaluation set is empty")
    hits = 0
    citation_hits = 0
    keyword_scores: list[float] = []
    details: list[dict[str, Any]] = []
    for case in cases:
        result = engine.answer(case["query"])
        retrieved_ids = {item["id"] for item in result.retrieved_documents}
        expected_ids = set(case.get("expected_doc_ids", []))
        hit = bool(retrieved_ids & expected_ids)
        required_terms = [term.lower() for term in case.get("required_terms", [])]
        answer_lower = result.answer.lower()
        keyword_score = (
            sum(term in answer_lower for term in required_terms) / len(required_terms)
            if required_terms else 1.0
        )
        hits += hit
        citation_hits += bool(result.citations)
        keyword_scores.append(keyword_score)
        details.append({
            "query": case["query"],
            "retrieval_hit": hit,
            "citation_present": bool(result.citations),
            "keyword_coverage": round(keyword_score, 4),
            "status": result.status,
        })
    count = len(cases)
    return {
        "cases": count,
        "retrieval_hit_rate": round(hits / count, 4),
        "citation_coverage": round(citation_hits / count, 4),
        "answer_keyword_coverage": round(sum(keyword_scores) / count, 4),
        "details": details,
    }
