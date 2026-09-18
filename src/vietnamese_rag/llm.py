from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


# Kept identical to the extractive fallback string in ``answering.py`` so both
# generation paths ("llm" and "extractive") return exactly the same sentence
# when there is not enough evidence, and downstream tests/eval don't need to
# special-case which generator produced the answer.
INSUFFICIENT_ANSWER_VI = (
    "Tôi chưa tìm thấy đủ bằng chứng trong cơ sở tri thức để trả lời câu hỏi này."
)

# Default OpenRouter model when ``OPENROUTER_MODEL`` is not set. This is the
# official OpenRouter auto-router: it selects a suitable free-tier model on
# OpenRouter's behalf and supports structured JSON output / tool calling, so
# it is a reasonable zero-config default. ``OPENROUTER_API_KEY`` has no such
# default -- it must still be provided explicitly (see ``build_llm_provider``).
DEFAULT_OPENROUTER_MODEL = "openrouter/free"

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class ContextChunk:
    """A single piece of retrieved evidence handed to the LLM.

    Deliberately independent from ``retrieval.SearchResult``/``Document`` so
    that ``llm.py`` has zero import-time dependency on the retrieval layer.
    ``answering.py`` is responsible for converting ``SearchResult`` objects
    into ``ContextChunk`` instances.
    """

    citation_id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class LLMAnswer:
    text: str
    citation_ids: list[str]
    insufficient: bool


class LLMError(RuntimeError):
    """Wraps any network/HTTP/parsing failure from an ``LLMProvider``.

    ``answering.py`` only needs to catch this single exception type to fall
    back to the extractive generator.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Common interface for any LLM-backed answer generator."""

    def generate(self, query: str, context: list[ContextChunk]) -> LLMAnswer:
        ...


SYSTEM_PROMPT_VI = f"""Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu tiếng Việt.

Quy tắc bắt buộc:
- Chỉ được sử dụng thông tin có trong phần NGỮ CẢNH bên dưới; tuyệt đối không dùng kiến thức bên ngoài NGỮ CẢNH.
- Luôn trả lời bằng tiếng Việt.
- Mỗi khẳng định trong câu trả lời phải kèm theo citation id lấy đúng từ nhãn [id] xuất hiện trong NGỮ CẢNH; không được tự tạo id mới không tồn tại trong NGỮ CẢNH.
- Nếu NGỮ CẢNH rỗng hoặc không liên quan tới câu hỏi, trường "answer" phải là đúng nguyên văn câu sau và "citations" phải là mảng rỗng:
  "{INSUFFICIENT_ANSWER_VI}"
- Chỉ được xuất ra một đối tượng JSON duy nhất, không thêm lời dẫn, không thêm markdown, theo đúng định dạng:
  {{"answer": "...", "citations": ["<id>", ...]}}
"""


def build_messages(query: str, context: list[ContextChunk]) -> list[dict]:
    """Render the system + user chat messages sent to the LLM.

    Known limitation: ``chunk.text`` (raw document content) is interpolated
    verbatim into the user message with no PII redaction and no sanitization
    against prompt injection. Any sensitive data or adversarial instructions
    present in the knowledge base are sent as-is to the configured
    third-party provider (OpenRouter) and are exposed to whatever model that
    provider routes the request to. Do not add untrusted or sensitive
    documents to the knowledge base without addressing this first.
    """
    if context:
        context_block = "\n\n".join(
            f"[{chunk.citation_id}] {chunk.title} (nguồn: {chunk.source})\n{chunk.text}"
            for chunk in context
        )
    else:
        context_block = "(rỗng)"

    user_content = f"NGỮ CẢNH:\n{context_block}\n\nCÂU HỎI: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT_VI},
        {"role": "user", "content": user_content},
    ]


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def parse_llm_response(raw: str, allowed_ids: set[str]) -> LLMAnswer:
    """Parse the raw model output into an ``LLMAnswer``.

    Tolerant of free-tier OpenRouter models that ignore
    ``response_format={"type": "json_object"}`` and wrap the JSON in a
    markdown code fence.
    """
    stripped = raw.strip() if isinstance(raw, str) else ""
    fence_match = _JSON_FENCE_PATTERN.search(stripped) if stripped else None
    candidate = fence_match.group(1) if fence_match else stripped

    # ``raw`` comes straight from the OpenRouter response body and is not
    # guaranteed to be a string (some malformed/edge-case responses return
    # ``content: null`` or a non-string value). Build the error preview via
    # ``str()`` -- which accepts any type -- instead of slicing ``raw``
    # directly, so a malformed response always raises ``LLMError`` (letting
    # ``GroundedAnswerEngine`` fall back to the extractive generator) rather
    # than an unrelated ``TypeError``.
    preview = str(raw)[:200]

    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMError(f"Could not parse LLM response as JSON: {preview!r}") from exc

    if not isinstance(payload, dict):
        raise LLMError(f"LLM response JSON is not an object: {preview!r}")

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise LLMError(f"LLM response missing non-empty 'answer' field: {preview!r}")

    raw_citations = payload.get("citations", [])
    if not isinstance(raw_citations, list):
        raw_citations = []

    citation_ids: list[str] = []
    for citation_id in raw_citations:
        if (
            isinstance(citation_id, str)
            and citation_id in allowed_ids
            and citation_id not in citation_ids
        ):
            citation_ids.append(citation_id)

    # ``allowed_ids`` empty means the caller passed no context chunks at all;
    # treat that the same as the model explicitly saying "not enough data".
    # The marker check uses "contains" rather than exact equality because
    # some models prepend/append extra wording (citations, apologies, etc.)
    # around the required sentence instead of returning it verbatim.
    insufficient = not allowed_ids or _normalize_whitespace(
        INSUFFICIENT_ANSWER_VI
    ) in _normalize_whitespace(answer)

    return LLMAnswer(text=answer, citation_ids=citation_ids, insufficient=insufficient)


class OpenRouterProvider:
    """``LLMProvider`` backed by OpenRouter's OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 800,
        client: httpx.Client | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY chưa được thiết lập")

        model = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def generate(self, query: str, context: list[ContextChunk]) -> LLMAnswer:
        messages = build_messages(query, context)
        allowed_ids = {chunk.citation_id for chunk in context}

        try:
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/khangkaka066/RAG-Vietnamese",
                    "X-Title": "Vietnamese RAG & LLM Evaluation Service",
                },
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"OpenRouter returned status {response.status_code}: {response.text[:200]}"
            )

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenRouter response shape: {exc}") from exc

        return parse_llm_response(content, allowed_ids)

    def close(self) -> None:
        """Release the underlying ``httpx.Client`` connection pool.

        Optional: the process-lifetime singleton built by
        ``build_llm_provider()`` (used by ``api.py``) is never closed
        explicitly, which is fine for a long-running service. Callers that
        create short-lived providers (e.g. scripts, tests with a real
        client) should call this when done.
        """
        self.client.close()

    def __enter__(self) -> "OpenRouterProvider":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_llm_provider() -> LLMProvider | None:
    """Factory that builds an ``OpenRouterProvider`` from environment variables.

    Returns ``None`` (a valid, offline-friendly outcome) when
    ``OPENROUTER_API_KEY`` is missing -- ``answering.py`` then falls back to
    the extractive generator. ``OPENROUTER_MODEL`` is optional: it defaults
    to ``DEFAULT_OPENROUTER_MODEL`` (OpenRouter's free auto-router) when
    unset.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        return None
    return OpenRouterProvider()
