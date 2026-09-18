from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
import json


TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode text while preserving Vietnamese words and accents."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    matched_terms: tuple[str, ...]


def load_documents(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            documents.append(Document(**record))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid document at line {line_number}") from exc
    if not documents:
        raise ValueError("Knowledge base is empty")
    return documents


@runtime_checkable
class Retriever(Protocol):
    """Common interface shared by lexical, embedding, and hybrid retrievers."""

    documents: list[Document]
    minimum_score: float

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        ...


class LexicalRetriever:
    """Small deterministic TF-IDF-style baseline for reproducible evaluation."""

    minimum_score: float = 0.15

    def __init__(self, documents: list[Document], minimum_score: float = 0.15) -> None:
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = documents
        self.minimum_score = minimum_score
        self._tokens = {doc.id: tokenize(f"{doc.title} {doc.text}") for doc in documents}
        document_frequency: dict[str, int] = {}
        for tokens in self._tokens.values():
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        count = len(documents)
        self._idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        ranked: list[SearchResult] = []
        for document in self.documents:
            tokens = self._tokens[document.id]
            frequencies = {token: tokens.count(token) for token in query_terms}
            matched = tuple(sorted(token for token, frequency in frequencies.items() if frequency))
            if not matched:
                continue
            raw_score = sum((1 + math.log(frequencies[token])) * self._idf[token] for token in matched)
            length_penalty = math.sqrt(max(len(tokens), 1))
            score = raw_score / length_penalty
            ranked.append(SearchResult(document, score, matched))
        ranked.sort(key=lambda item: (-item.score, item.document.id))
        return ranked[: max(top_k, 1)]


class HybridRetriever:
    """Combine a lexical and an embedding retriever on an absolute score scale.

    For ``0 < alpha < 1``: ``score = alpha * cosine + (1 - alpha) * min(lexical_score, 1.0)``.
    The score is intentionally NOT min-max normalized per query: normalizing would
    always push the best candidate to 1.0, even for a completely irrelevant query,
    which would break the ``UNAVAILABLE`` behaviour driven by ``minimum_score``.

    At the boundaries (``alpha == 0.0`` or ``alpha == 1.0``), the fusion formula is
    bypassed entirely and the underlying retriever's own ``search`` results are
    returned unchanged (see below). This guarantees ``alpha=0.0`` is *exactly*
    lexical-only and ``alpha=1.0`` is *exactly* embedding-only, including tie-break
    order — capping/blending lexical's unbounded score against embedding's bounded
    ``[-1, 1]`` cosine range at those extremes would otherwise destroy ranking
    information (e.g. two lexical scores of 1.97 and 1.85 both collapsing to 1.0).
    """

    def __init__(
        self,
        lexical: "Retriever",
        embedding: "Retriever",
        alpha: float = 0.5,
        minimum_score: float | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be within [0.0, 1.0], got {alpha!r}")
        self.lexical = lexical
        self.embedding = embedding
        self.alpha = alpha
        self.documents = lexical.documents
        if minimum_score is not None:
            self.minimum_score = minimum_score
        else:
            self.minimum_score = alpha * getattr(embedding, "minimum_score", 0.3) + (
                1 - alpha
            ) * min(getattr(lexical, "minimum_score", 0.15), 1.0)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            return []

        # Bypass the fusion formula entirely at the two boundary values. Lexical
        # scores are unbounded (a document can score well above 1.0 when it
        # matches several rare/high-IDF terms), so any attempt to combine or cap
        # them alongside the embedding branch's [-1, 1] cosine range would either
        # clip away real ranking information or require a normalization scheme
        # that risks breaking ties differently than the underlying retriever.
        # Returning the underlying retriever's own results directly is the only
        # way to *guarantee* alpha=0.0 == lexical-only and alpha=1.0 ==
        # embedding-only, exactly, including tie-break order.
        if self.alpha == 0.0:
            return self.lexical.search(query, top_k=top_k)
        if self.alpha == 1.0:
            return self.embedding.search(query, top_k=top_k)

        # Score every document on both branches (not just a top-N pool). Both
        # LexicalRetriever and EmbeddingRetriever already rank their full
        # ``self.documents`` internally before truncating to ``top_k``, so asking
        # for ``len(self.documents)`` results simply removes that truncation and
        # gives us the *exact* score each branch would assign to every candidate.
        # This is required for the alpha=1.0 == embedding-only and alpha=0.0 ==
        # lexical-only contract to hold: a top-N union pool can silently drop a
        # document from one branch (e.g. when cosine similarity is negative, or
        # when a document only matches lexically), which previously made the
        # missing branch fall back to a hardcoded 0.0 and corrupted the ranking.
        pool_size = len(self.documents)
        lexical_results = self.lexical.search(query, top_k=pool_size)
        embedding_results = self.embedding.search(query, top_k=pool_size)

        lexical_by_id = {result.document.id: result for result in lexical_results}
        embedding_by_id = {result.document.id: result for result in embedding_results}

        # EmbeddingRetriever always scores every document, so embedding_by_id is
        # guaranteed to be complete. LexicalRetriever, by design, drops documents
        # that share no query term at all; for those, 0.0 correctly represents
        # "no lexical evidence" since LexicalRetriever's own scores are bounded
        # at (or above) its minimum_score for anything it does return.
        candidate_ids = set(lexical_by_id) | set(embedding_by_id)
        ranked: list[SearchResult] = []
        for document_id in candidate_ids:
            lexical_result = lexical_by_id.get(document_id)
            embedding_result = embedding_by_id.get(document_id)
            document = (lexical_result or embedding_result).document
            lexical_score = min(lexical_result.score, 1.0) if lexical_result else 0.0
            cosine_score = embedding_result.score if embedding_result else 0.0
            score = self.alpha * cosine_score + (1 - self.alpha) * lexical_score
            matched_terms = lexical_result.matched_terms if lexical_result else ()
            ranked.append(SearchResult(document, score, matched_terms))

        ranked.sort(key=lambda item: (-item.score, item.document.id))
        return ranked[: max(top_k, 1)]


def build_retriever(documents: list[Document], mode: str | None = None, **kwargs) -> "Retriever":
    """Factory that picks a retriever implementation.

    ``mode`` defaults to the ``VIETNAMESE_RETRIEVER`` environment variable, which in turn
    defaults to ``"lexical"``. This keeps the system-wide default fully offline
    (no ``sentence-transformers`` dependency required) while allowing embedding or
    hybrid retrieval to be enabled explicitly.
    """
    resolved_mode = mode or os.environ.get("VIETNAMESE_RETRIEVER", "lexical")

    if resolved_mode == "lexical":
        return LexicalRetriever(documents, **kwargs)

    if resolved_mode == "embedding":
        from .retrieval_embedding import EmbeddingRetriever

        return EmbeddingRetriever(documents, **kwargs)

    if resolved_mode == "hybrid":
        from .retrieval_embedding import EmbeddingRetriever

        alpha = kwargs.pop("alpha", 0.5)
        minimum_score = kwargs.pop("minimum_score", None)
        lexical = LexicalRetriever(documents)
        embedding = EmbeddingRetriever(documents, **kwargs)
        return HybridRetriever(lexical, embedding, alpha=alpha, minimum_score=minimum_score)

    raise ValueError(
        f"Unknown retriever mode {resolved_mode!r}; expected one of "
        "'lexical', 'embedding', 'hybrid'"
    )
