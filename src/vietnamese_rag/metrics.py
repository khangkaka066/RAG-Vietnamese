"""Pure, offline scoring functions used by ``evaluation.evaluate``.

No imports from ``answering``/``llm`` on purpose, to avoid a dependency
cycle with ``evaluation.py``: only ``tokenize``/``Document`` from
``retrieval.py`` are needed.
"""

from __future__ import annotations

import math

from .retrieval import Document, tokenize


def build_idf(documents: list[Document]) -> dict[str, float]:
    """Corpus-wide IDF over ``title + text``, matching ``LexicalRetriever``."""
    document_frequency: dict[str, int] = {}
    for document in documents:
        for token in set(tokenize(f"{document.title} {document.text}")):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    count = len(documents)
    return {
        token: math.log((count + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }


def default_idf(documents: list[Document]) -> float:
    """IDF assigned to a token absent from the corpus vocabulary.

    Treated as maximally informative (higher than any in-vocabulary token),
    so an out-of-context token is penalized heavily -- matching the intent
    of "penalize made-up information not grounded in the context".
    """
    return math.log(len(documents) + 1) + 1


def idf_weighted_containment(candidate: str, reference: str, idf: dict[str, float], oov_idf: float) -> float:
    """Fraction of ``candidate``'s (IDF-weighted) token mass also found in ``reference``.

    Returns a value in ``[0, 1]``; ``0.0`` when ``candidate`` has no tokens.
    """
    candidate_tokens = set(tokenize(candidate))
    if not candidate_tokens:
        return 0.0
    reference_tokens = set(tokenize(reference))
    weights = {token: idf.get(token, oov_idf) for token in candidate_tokens}
    denominator = sum(weights.values())
    if denominator == 0.0:
        return 0.0
    numerator = sum(weight for token, weight in weights.items() if token in reference_tokens)
    return numerator / denominator


def faithfulness(answer: str, context_text: str, idf: dict[str, float], oov_idf: float) -> float:
    """How much of the answer's content is grounded in the retrieved context."""
    return idf_weighted_containment(answer, context_text, idf, oov_idf)


def answer_relevancy(query: str, answer: str, idf: dict[str, float], oov_idf: float) -> float:
    """How much of the query's content is addressed by the answer."""
    return idf_weighted_containment(query, answer, idf, oov_idf)


def mrr_at_k(ranked_ids: list[str], expected_ids: set[str], k: int) -> tuple[float, int | None]:
    """Reciprocal rank of the first expected id within the top ``k`` of ``ranked_ids``.

    Returns ``(reciprocal_rank, rank)`` where ``rank`` is the 1-based
    position of the hit, or ``None`` when no expected id appears in the
    top ``k``.
    """
    for position, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in expected_ids:
            return 1.0 / position, position
    return 0.0, None


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in ``[0, 100]``), no numpy dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
