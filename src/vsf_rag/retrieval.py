from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
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


class LexicalRetriever:
    """Small deterministic TF-IDF-style baseline for reproducible evaluation."""

    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = documents
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
