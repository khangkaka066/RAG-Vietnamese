"""Compare hit-rate/MRR@3 of lexical, embedding, and hybrid retrievers.

Requires the optional ``embeddings`` extra (``pip install -e '.[embeddings]'``).
Not run in CI: this script downloads and runs the sentence-transformers model.
"""
from __future__ import annotations

from pathlib import Path

from vietnamese_rag.retrieval import LexicalRetriever, HybridRetriever, load_documents
from vietnamese_rag.retrieval_embedding import EmbeddingRetriever
from vietnamese_rag.evaluation import load_eval_cases


ROOT = Path(__file__).resolve().parents[1]


def mrr_at_k(retriever, cases: list[dict], k: int = 3) -> float:
    reciprocal_ranks = []
    for case in cases:
        expected = set(case.get("expected_doc_ids", []))
        results = retriever.search(case["query"], top_k=k)
        rank = next(
            (i + 1 for i, result in enumerate(results) if result.document.id in expected),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def hit_rate_at_k(retriever, cases: list[dict], k: int = 3) -> float:
    hits = 0
    for case in cases:
        expected = set(case.get("expected_doc_ids", []))
        results = retriever.search(case["query"], top_k=k)
        hits += bool({result.document.id for result in results} & expected)
    return hits / len(cases)


def main() -> None:
    documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
    cases = load_eval_cases(ROOT / "data" / "eval.jsonl")

    lexical = LexicalRetriever(documents)
    embedding = EmbeddingRetriever(documents)
    hybrid = HybridRetriever(lexical, embedding)

    print(f"{'retriever':<12}{'hit_rate@3':>12}{'mrr@3':>10}")
    for name, retriever in [("lexical", lexical), ("embedding", embedding), ("hybrid", hybrid)]:
        hit_rate = hit_rate_at_k(retriever, cases)
        mrr = mrr_at_k(retriever, cases)
        print(f"{name:<12}{hit_rate:>12.4f}{mrr:>10.4f}")


if __name__ == "__main__":
    main()
