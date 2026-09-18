from __future__ import annotations

import math
from typing import Callable, Sequence

from .retrieval import Document, SearchResult, tokenize


DEFAULT_EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

Encoder = Callable[[list[str]], list[Sequence[float]]]


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two already-L2-normalized vectors, i.e. their cosine similarity."""
    if len(a) != len(b):
        raise ValueError(
            f"Cannot compute cosine similarity between vectors of different dimensions: "
            f"{len(a)} != {len(b)}"
        )
    return sum(x * y for x, y in zip(a, b))


def _default_encoder(model_name: str) -> Encoder:
    """Build an encoder backed by ``sentence-transformers``.

    Imported lazily so the rest of the codebase (and CI, by default) never needs
    ``sentence-transformers``/``torch`` installed. Raises a clear, actionable error if the
    optional dependency is missing.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for EmbeddingRetriever. "
            "Install it with: pip install -e '.[embeddings]'"
        ) from exc

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> list[Sequence[float]]:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [[float(component) for component in vector] for vector in embeddings]

    return encode


class EmbeddingRetriever:
    """Cosine-similarity retriever over Vietnamese sentence embeddings.

    ``encoder`` is a dependency-injection point: a callable ``(list[str]) ->
    list[Sequence[float]]``. When ``None``, a ``SentenceTransformer`` for
    ``model_name`` is lazily constructed inside ``__init__``. This lets tests exercise
    the cosine/top-k logic offline with a fake encoder, without requiring torch.
    """

    def __init__(
        self,
        documents: list[Document],
        encoder: Encoder | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        # NOTE: the plan originally proposed 0.5 as a starting guess before any
        # empirical measurement. Measured cosine distribution on the real
        # bkai-foundation-models/vietnamese-bi-encoder model against
        # data/knowledge_base.jsonl + data/eval.jsonl showed out-of-domain
        # queries topping out at cosine ~0.2063 while in-domain queries never
        # dropped below ~0.3364, i.e. a clear separation gap around 0.3. Using
        # 0.5 would incorrectly reject at least one valid in-domain case
        # (eval-003). 0.3 sits inside the observed gap and is used instead.
        minimum_score: float = 0.3,
    ) -> None:
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = documents
        self.minimum_score = minimum_score
        self._encoder: Encoder = encoder or _default_encoder(model_name)

        texts = [f"{document.title} {document.text}" for document in documents]
        raw_vectors = self._encoder(texts)
        if len(raw_vectors) != len(documents):
            raise ValueError(
                f"Encoder returned {len(raw_vectors)} vectors for {len(documents)} documents; "
                "expected exactly one vector per document"
            )
        self._vectors: dict[str, list[float]] = {
            document.id: _l2_normalize(vector)
            for document, vector in zip(documents, raw_vectors, strict=True)
        }
        vector_dimensions = {len(vector) for vector in self._vectors.values()}
        if len(vector_dimensions) > 1:
            raise ValueError(
                f"Encoder returned vectors of inconsistent dimensions: {sorted(vector_dimensions)}"
            )
        self._tokens = {doc.id: set(tokenize(f"{doc.title} {doc.text}")) for doc in documents}

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            return []

        query_vector = _l2_normalize(self._encoder([query])[0])
        query_terms = set(tokenize(query))

        ranked: list[SearchResult] = []
        for document in self.documents:
            score = _cosine(query_vector, self._vectors[document.id])
            matched_terms = tuple(sorted(query_terms & self._tokens[document.id]))
            ranked.append(SearchResult(document, score, matched_terms))

        ranked.sort(key=lambda item: (-item.score, item.document.id))
        return ranked[: max(top_k, 1)]
