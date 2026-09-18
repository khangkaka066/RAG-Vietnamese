from __future__ import annotations

import json

import httpx
import pytest

from vietnamese_rag.llm import (
    ContextChunk,
    DEFAULT_OPENROUTER_MODEL,
    INSUFFICIENT_ANSWER_VI,
    LLMError,
    OpenRouterProvider,
    build_llm_provider,
    build_messages,
    parse_llm_response,
)


CONTEXT = [
    ContextChunk(citation_id="rag-001", title="RAG principles", source="internal-demo/rag-principles.md", text="RAG cần trích dẫn nguồn."),
    ContextChunk(citation_id="rag-002", title="Citations", source="internal-demo/citations.md", text="Citation giúp kiểm chứng câu trả lời."),
]


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_contains_every_citation_id_and_insufficient_marker() -> None:
    messages = build_messages("RAG cần trích dẫn nguồn như thế nào?", CONTEXT)

    assert messages[0]["role"] == "system"
    assert INSUFFICIENT_ANSWER_VI in messages[0]["content"]

    assert messages[1]["role"] == "user"
    for chunk in CONTEXT:
        assert chunk.citation_id in messages[1]["content"]
        assert chunk.text in messages[1]["content"]


def test_build_messages_with_empty_context_still_produces_user_message() -> None:
    messages = build_messages("query lạ", [])
    assert messages[1]["role"] == "user"
    assert "query lạ" in messages[1]["content"]


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------


def test_parse_llm_response_parses_plain_json() -> None:
    raw = json.dumps({"answer": "RAG cần trích dẫn nguồn.", "citations": ["rag-001"]})
    result = parse_llm_response(raw, allowed_ids={"rag-001", "rag-002"})
    assert result.text == "RAG cần trích dẫn nguồn."
    assert result.citation_ids == ["rag-001"]
    assert result.insufficient is False


def test_parse_llm_response_parses_json_wrapped_in_markdown_fence() -> None:
    raw = "```json\n" + json.dumps({"answer": "Câu trả lời.", "citations": ["rag-001"]}) + "\n```"
    result = parse_llm_response(raw, allowed_ids={"rag-001"})
    assert result.text == "Câu trả lời."
    assert result.citation_ids == ["rag-001"]


def test_parse_llm_response_filters_unknown_citation_ids_and_dedupes() -> None:
    raw = json.dumps({"answer": "OK", "citations": ["rag-001", "unknown-id", "rag-001", "rag-002"]})
    result = parse_llm_response(raw, allowed_ids={"rag-001", "rag-002"})
    assert result.citation_ids == ["rag-001", "rag-002"]


def test_parse_llm_response_marks_insufficient_when_answer_matches_marker() -> None:
    raw = json.dumps({"answer": INSUFFICIENT_ANSWER_VI, "citations": []})
    result = parse_llm_response(raw, allowed_ids={"rag-001"})
    assert result.insufficient is True
    assert result.citation_ids == []


def test_parse_llm_response_marks_insufficient_when_marker_has_surrounding_text() -> None:
    # Some models wrap the required marker sentence with extra wording
    # (apologies, citations, ...) instead of returning it verbatim.
    raw = json.dumps(
        {"answer": f"Xin lỗi, {INSUFFICIENT_ANSWER_VI} [rag-001]", "citations": ["rag-001"]}
    )
    result = parse_llm_response(raw, allowed_ids={"rag-001"})
    assert result.insufficient is True


def test_parse_llm_response_marks_insufficient_when_no_context_available() -> None:
    raw = json.dumps({"answer": "Bất kỳ câu trả lời nào.", "citations": []})
    result = parse_llm_response(raw, allowed_ids=set())
    assert result.insufficient is True


def test_parse_llm_response_raises_on_invalid_json() -> None:
    with pytest.raises(LLMError):
        parse_llm_response("khong phai json", allowed_ids={"rag-001"})


def test_parse_llm_response_raises_when_answer_field_missing() -> None:
    with pytest.raises(LLMError):
        parse_llm_response(json.dumps({"citations": ["rag-001"]}), allowed_ids={"rag-001"})


def test_parse_llm_response_raises_when_answer_field_empty() -> None:
    with pytest.raises(LLMError):
        parse_llm_response(json.dumps({"answer": "  ", "citations": []}), allowed_ids={"rag-001"})


@pytest.mark.parametrize("raw", [None, 5, 3.14, ["not", "a", "string"], {"content": "nested"}])
def test_parse_llm_response_raises_llm_error_on_non_string_content(raw: object) -> None:
    # Regression test for Codex review round 2, finding #3: OpenRouter can
    # return a malformed body where ``choices[0].message.content`` is not a
    # string (e.g. ``null``, a number, a list, ...). Building the error
    # preview used to slice ``raw`` directly (``raw[:200]``), which raised
    # ``TypeError``/``KeyError`` instead of the expected ``LLMError`` --
    # breaking the extractive fallback in ``GroundedAnswerEngine``.
    with pytest.raises(LLMError):
        parse_llm_response(raw, allowed_ids={"rag-001"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OpenRouterProvider
# ---------------------------------------------------------------------------


def _make_provider(handler, **kwargs) -> OpenRouterProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://openrouter.ai/api/v1", transport=transport)
    return OpenRouterProvider(api_key="test-key", model="test-model", client=client, **kwargs)


def test_openrouter_provider_sends_expected_request_and_parses_answer() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["json"] = json.loads(request.content)
        payload = {
            "choices": [
                {"message": {"content": json.dumps({"answer": "Trả lời.", "citations": ["rag-001"]})}}
            ]
        }
        return httpx.Response(200, json=payload, request=request)

    provider = _make_provider(handler)
    result = provider.generate("câu hỏi?", CONTEXT)

    assert result.text == "Trả lời."
    assert result.citation_ids == ["rag-001"]

    sent = captured["json"]
    assert sent["model"] == "test-model"
    assert sent["temperature"] == 0.0
    assert sent["response_format"] == {"type": "json_object"}
    assert captured["request"].headers["authorization"] == "Bearer test-key"


def test_openrouter_provider_raises_llm_error_on_http_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited", request=request)

    provider = _make_provider(handler)
    with pytest.raises(LLMError):
        provider.generate("câu hỏi?", CONTEXT)


def test_openrouter_provider_raises_llm_error_on_malformed_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"}, request=request)

    provider = _make_provider(handler)
    with pytest.raises(LLMError):
        provider.generate("câu hỏi?", CONTEXT)


def test_openrouter_provider_requires_api_key() -> None:
    with pytest.raises(LLMError):
        OpenRouterProvider(api_key=None, model="test-model")


def test_openrouter_provider_defaults_model_when_env_var_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    provider = OpenRouterProvider(api_key="test-key")
    assert provider.model == DEFAULT_OPENROUTER_MODEL


# ---------------------------------------------------------------------------
# build_llm_provider
# ---------------------------------------------------------------------------


def test_build_llm_provider_returns_none_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert build_llm_provider() is None


def test_build_llm_provider_returns_provider_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    provider = build_llm_provider()
    assert isinstance(provider, OpenRouterProvider)
    assert provider.api_key == "env-key"
    assert provider.model == DEFAULT_OPENROUTER_MODEL


def test_build_llm_provider_uses_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model:free")
    provider = build_llm_provider()
    assert provider.model == "some/model:free"
