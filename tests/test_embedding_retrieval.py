from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from vsf_rag.answering import GroundedAnswerEngine
from vsf_rag.evaluation import evaluate, load_eval_cases
from vsf_rag.retrieval import Document, HybridRetriever, LexicalRetriever, build_retriever, load_documents
from vsf_rag.retrieval_embedding import EmbeddingRetriever


ROOT = Path(__file__).resolve().parents[1]

FAKE_VOCAB = ["rag", "trích", "dẫn", "context", "faithfulness", "độ", "trễ", "schema", "timeout", "log"]


def fake_encoder(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words style encoder for offline unit tests.

    No torch/sentence-transformers required: it just counts, for each text, how many
    times each token in a fixed vocabulary occurs.
    """
    from vsf_rag.retrieval import tokenize

    vectors: list[list[float]] = []
    for text in texts:
        tokens = tokenize(text)
        vectors.append([float(tokens.count(term)) for term in FAKE_VOCAB])
    return vectors


def build_fake_documents() -> list[Document]:
    return [
        Document(id="a", title="RAG", source="a.md", text="RAG cần trích dẫn context rõ ràng"),
        Document(id="b", title="Eval", source="b.md", text="faithfulness và độ trễ là chỉ số quan trọng"),
        Document(id="c", title="Agent", source="c.md", text="schema và timeout cần log lại đầy đủ"),
    ]


# ---------------------------------------------------------------------------
# Group A: offline, deterministic, fake encoder (no torch dependency)
# ---------------------------------------------------------------------------


def test_cosine_is_bounded_and_one_for_identical_vector() -> None:
    documents = build_fake_documents()
    retriever = EmbeddingRetriever(documents, encoder=fake_encoder)

    # EmbeddingRetriever encodes documents as f"{title} {text}", so the query must match
    # that same concatenation to reproduce an identical vector (cosine == 1.0).
    query = f"{documents[0].title} {documents[0].text}"
    results = retriever.search(query, top_k=3)
    assert all(-1.0 - 1e-9 <= result.score <= 1.0 + 1e-9 for result in results)
    assert results[0].document.id == "a"
    assert math.isclose(results[0].score, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_topk_order_and_tie_break_by_document_id() -> None:
    documents = [
        Document(id="z", title="Dup", source="z.md", text="RAG cần trích dẫn context rõ ràng"),
        Document(id="a", title="Dup", source="a.md", text="RAG cần trích dẫn context rõ ràng"),
    ]
    retriever = EmbeddingRetriever(documents, encoder=fake_encoder)
    results = retriever.search("RAG cần trích dẫn context rõ ràng", top_k=2)
    # identical scores -> tie-break by document id, matching LexicalRetriever's behaviour
    assert [result.document.id for result in results] == ["a", "z"]


def test_empty_query_returns_no_results() -> None:
    retriever = EmbeddingRetriever(build_fake_documents(), encoder=fake_encoder)
    assert retriever.search("   ", top_k=3) == []


def test_top_k_zero_still_returns_one_result() -> None:
    retriever = EmbeddingRetriever(build_fake_documents(), encoder=fake_encoder)
    results = retriever.search("RAG cần trích dẫn context", top_k=0)
    assert len(results) == 1


def test_encoder_returning_wrong_vector_count_raises() -> None:
    def short_encoder(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts[:-1]]  # one vector short

    with pytest.raises(ValueError):
        EmbeddingRetriever(build_fake_documents(), encoder=short_encoder)


def test_encoder_returning_inconsistent_dimensions_raises() -> None:
    def ragged_encoder(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if i % 2 == 0 else [1.0, 0.0, 0.0] for i, _ in enumerate(texts)]

    with pytest.raises(ValueError):
        EmbeddingRetriever(build_fake_documents(), encoder=ragged_encoder)


def test_query_vector_dimension_mismatch_raises() -> None:
    """Query encoded to a different dimension than the documents must raise, not silently truncate."""
    calls = {"n": 0}

    def inconsistent_query_encoder(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        if calls["n"] == 1:
            return [[1.0, 0.0] for _ in texts]  # document encoding pass
        return [[1.0, 0.0, 0.0]]  # query encoding pass: wrong dimension

    retriever = EmbeddingRetriever(build_fake_documents(), encoder=inconsistent_query_encoder)
    with pytest.raises(ValueError):
        retriever.search("RAG cần trích dẫn context", top_k=3)


def test_hybrid_alpha_one_matches_embedding_only_ranking() -> None:
    documents = build_fake_documents()
    lexical = LexicalRetriever(documents)
    embedding = EmbeddingRetriever(documents, encoder=fake_encoder)
    hybrid = HybridRetriever(lexical, embedding, alpha=1.0)

    query = "faithfulness và độ trễ"
    hybrid_ids = [result.document.id for result in hybrid.search(query, top_k=3)]
    embedding_ids = [result.document.id for result in embedding.search(query, top_k=3)]
    assert hybrid_ids == embedding_ids


def test_hybrid_alpha_zero_matches_lexical_only_ranking() -> None:
    documents = build_fake_documents()
    lexical = LexicalRetriever(documents)
    embedding = EmbeddingRetriever(documents, encoder=fake_encoder)
    hybrid = HybridRetriever(lexical, embedding, alpha=0.0)

    # Use a query that every document matches lexically (one distinguishing term each)
    # so the comparison covers the full ranking (LexicalRetriever drops non-matching
    # documents entirely) while keeping each raw score close to/below 1.0, matching the
    # scale HybridRetriever clips lexical scores to.
    query = "rag faithfulness schema"
    hybrid_ids = [result.document.id for result in hybrid.search(query, top_k=3)]
    lexical_ids = [result.document.id for result in lexical.search(query, top_k=3)]
    assert hybrid_ids == lexical_ids


def test_hybrid_alpha_one_matches_embedding_only_with_negative_cosine_and_many_documents() -> None:
    """Adversarial regression test for the alpha=1.0 == embedding-only contract.

    Reproduces the bug found in review: HybridRetriever used to fetch only a
    top-N pool (N derived from top_k) from each branch and default a document's
    score on a branch to 0.0 whenever it fell outside that branch's pool. With
    more documents than the pool size and a document whose cosine similarity is
    strongly *negative* (but which happens to match lexically, so it survives
    the lexical pool), the buggy default of 0.0 made it outrank real, correctly
    scored embedding candidates -- corrupting the alpha=1.0 ranking even though
    the lexical branch should have zero influence at alpha=1.0.
    """
    query_text = "consulta especial alpha"

    embedding_vectors: dict[str, list[float]] = {}
    documents: list[Document] = []

    # Best match by embedding, no lexical overlap with the query at all.
    documents.append(Document(id="embed-best", title="E", source="e.md", text="zzz yyy xxx"))
    embedding_vectors["E zzz yyy xxx"] = [1.0, 0.0]

    # 11 filler documents: mildly *negative* cosine, no lexical overlap. There are
    # enough of them that the total document count exceeds any small top-N pool
    # (the old code used max(top_k * 4, 10), i.e. 12 for top_k=3).
    for i in range(11):
        text = f"filler topic number {i}"
        documents.append(Document(id=f"filler-{i}", title="F", source=f"f{i}.md", text=text))
        embedding_vectors[f"F {text}"] = [-0.05, 0.0]

    # The adversarial document: matches the query lexically (so it always shows
    # up in the lexical branch's results) but has the *most negative* cosine of
    # the whole corpus, so it is the worst possible embedding-branch candidate
    # and is the one dropped when only a top-N pool is requested.
    documents.append(
        Document(id="lexical-match-worst-embedding", title="L", source="l.md", text="consulta especial alpha")
    )
    embedding_vectors["L consulta especial alpha"] = [-1.0, 0.0]

    embedding_vectors["consulta especial alpha"] = [1.0, 0.0]  # query vector

    def adversarial_encoder(texts: list[str]) -> list[list[float]]:
        return [embedding_vectors[text] for text in texts]

    assert len(documents) == 13  # sanity: bigger than the old fixed top-N pool

    lexical = LexicalRetriever(documents)
    embedding = EmbeddingRetriever(documents, encoder=adversarial_encoder)
    hybrid = HybridRetriever(lexical, embedding, alpha=1.0)

    top_k = 3
    hybrid_ids = [result.document.id for result in hybrid.search(query_text, top_k=top_k)]
    embedding_ids = [result.document.id for result in embedding.search(query_text, top_k=top_k)]

    assert hybrid_ids == embedding_ids
    assert embedding_ids[0] == "embed-best"
    # The adversarial document must never be promoted above genuinely better
    # (less negative) embedding candidates just because it also matches lexically.
    assert "lexical-match-worst-embedding" not in hybrid_ids

    # The full ranking must also put the adversarial document dead last: its
    # true cosine similarity (-1.0) is worse than every filler (-0.05).
    full_embedding_ids = [
        result.document.id for result in embedding.search(query_text, top_k=len(documents))
    ]
    assert full_embedding_ids[-1] == "lexical-match-worst-embedding"


def test_hybrid_alpha_zero_matches_lexical_only_ranking_above_unit_score() -> None:
    """Adversarial regression test for the alpha=0.0 == lexical-only contract.

    Reproduces the bug found in review: the fusion path used to clip every
    lexical score with ``min(lexical_score, 1.0)`` before blending. At
    alpha=0.0 that clip has no mathematical reason to run at all (the
    embedding branch contributes nothing), yet it used to still apply,
    collapsing any two documents whose raw lexical scores both exceed 1.0
    down to the same 1.0 and silently falling back to document-id tie-break
    -- which can reorder documents relative to genuine lexical-only ranking.

    These three documents are built so two of them score *above* 1.0 (e.g.
    ~1.07 and ~1.06 here), with the higher-scoring document ("mid") sorting
    *after* the lower-scoring one ("hi") by document id -- the exact
    ordering that a min(..., 1.0) clip + id tie-break would get wrong.
    """
    documents = [
        Document(id="hi", title="H", source="h.md", text="alpha alpha alpha alpha alpha alpha"),
        Document(id="mid", title="M", source="m.md", text="alpha alpha alpha alpha"),
        Document(id="lo", title="L", source="l.md", text="alpha"),
    ]
    lexical = LexicalRetriever(documents)
    embedding = EmbeddingRetriever(documents, encoder=fake_encoder)
    hybrid = HybridRetriever(lexical, embedding, alpha=0.0)

    lexical_results = lexical.search("alpha", top_k=3)
    # Sanity-check the adversarial setup: at least two raw scores must exceed
    # 1.0, and the higher-scoring one must sort *after* the lower-scoring one
    # by document id, so a min(score, 1.0) clip + id tie-break would corrupt
    # the order.
    scores_by_id = {result.document.id: result.score for result in lexical_results}
    assert scores_by_id["mid"] > 1.0
    assert scores_by_id["hi"] > 1.0
    assert scores_by_id["mid"] > scores_by_id["hi"]
    assert "hi" < "mid"  # "hi" would incorrectly sort first under id tie-break

    hybrid_ids = [result.document.id for result in hybrid.search("alpha", top_k=3)]
    lexical_ids = [result.document.id for result in lexical_results]
    assert hybrid_ids == lexical_ids == ["mid", "hi", "lo"]


def test_hybrid_rejects_alpha_outside_unit_interval() -> None:
    documents = build_fake_documents()
    lexical = LexicalRetriever(documents)
    embedding = EmbeddingRetriever(documents, encoder=fake_encoder)

    with pytest.raises(ValueError):
        HybridRetriever(lexical, embedding, alpha=1.5)
    with pytest.raises(ValueError):
        HybridRetriever(lexical, embedding, alpha=-0.1)


def test_engine_uses_retriever_minimum_score_by_default() -> None:
    documents = build_fake_documents()
    embedding = EmbeddingRetriever(documents, encoder=fake_encoder, minimum_score=0.9)
    engine = GroundedAnswerEngine(embedding)
    assert engine.minimum_score == 0.9

    engine_override = GroundedAnswerEngine(embedding, minimum_score=0.2)
    assert engine_override.minimum_score == 0.2


def test_build_retriever_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        build_retriever(build_fake_documents(), mode="not-a-real-mode")


def test_build_retriever_defaults_to_lexical() -> None:
    retriever = build_retriever(build_fake_documents())
    assert isinstance(retriever, LexicalRetriever)


# ---------------------------------------------------------------------------
# Group B: real sentence-transformers model, hit-rate comparison (skipped when
# the optional dependency or the model download is unavailable).
# ---------------------------------------------------------------------------


def _real_embedding_retriever_or_skip(documents):
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")

    # Only skip for errors that indicate the *environment* can't load the model
    # (no network access, no local cache, etc). Anything else (e.g. a ValueError
    # from a real bug in EmbeddingRetriever's own validation logic) must be left
    # to fail the test loudly instead of being silently skipped.
    from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

    try:
        return EmbeddingRetriever(documents)
    except (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OSError,
        ConnectionError,
        TimeoutError,
    ) as exc:  # pragma: no cover - depends on network/model availability
        pytest.skip(f"could not load embedding model: {exc}")


@pytest.fixture(scope="module")
def real_embedding_retriever():
    documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
    return _real_embedding_retriever_or_skip(documents)


def test_embedding_and_hybrid_hit_rate_vs_lexical(real_embedding_retriever) -> None:
    documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
    eval_cases = load_eval_cases(ROOT / "data" / "eval.jsonl")

    lexical = LexicalRetriever(documents)
    embedding = real_embedding_retriever
    hybrid = HybridRetriever(lexical, embedding)

    lexical_report = evaluate(GroundedAnswerEngine(lexical), eval_cases)
    embedding_report = evaluate(GroundedAnswerEngine(embedding), eval_cases)
    hybrid_report = evaluate(GroundedAnswerEngine(hybrid), eval_cases)

    print("lexical_hit_rate", lexical_report["retrieval_hit_rate"])
    print("embedding_hit_rate", embedding_report["retrieval_hit_rate"])
    print("hybrid_hit_rate", hybrid_report["retrieval_hit_rate"])

    assert embedding_report["retrieval_hit_rate"] >= 0.8
    assert hybrid_report["retrieval_hit_rate"] >= (
        max(lexical_report["retrieval_hit_rate"], embedding_report["retrieval_hit_rate"]) - 0.05
    )


def test_out_of_domain_query_is_unavailable_for_all_modes(real_embedding_retriever) -> None:
    documents = load_documents(ROOT / "data" / "knowledge_base.jsonl")
    lexical = LexicalRetriever(documents)
    embedding = real_embedding_retriever
    hybrid = HybridRetriever(lexical, embedding)

    query = "Thời tiết hôm nay ở sao Hỏa thế nào?"
    for retriever in (lexical, embedding, hybrid):
        engine = GroundedAnswerEngine(retriever)
        result = engine.answer(query)
        assert result.status == "UNAVAILABLE", type(retriever).__name__
