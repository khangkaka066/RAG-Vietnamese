from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RouteDecision:
    """Result of classifying a query as "answer via RAG" or "call a tool"."""

    route: str  # "rag" | "tool"
    reason: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    router: str = "rule"


class Router(Protocol):
    def route(self, query: str) -> RouteDecision:
        ...


# Vietnamese/English phrases that strongly signal "the user wants the current
# date/time", checked against the lowercased query.
_DATETIME_KEYWORDS = (
    "hôm nay",
    "ngày mấy",
    "bây giờ là mấy giờ",
    "mấy giờ",
    "ngày giờ hiện tại",
    "today",
    "current time",
    "current date",
)

# Natural-language prefixes/suffixes stripped off an arithmetic query before
# checking whether what remains is a pure expression.
_ARITHMETIC_AFFIXES = (
    "tính giúp",
    "tính",
    "bằng bao nhiêu",
    "=",
    "?",
)

# After stripping affixes, the remainder must consist *only* of digits,
# whitespace, parentheses and arithmetic operators -- and contain at least
# one operator, so bare numbers ("2024") or section references ("điều 5")
# are not misdetected as arithmetic.
_ARITHMETIC_CHARS_RE = re.compile(r"^[0-9\.\s()+\-*/%^]+$")
_ARITHMETIC_OPERATOR_RE = re.compile(r"[+\-*/%^]")


def _extract_arithmetic_expression(query: str) -> str | None:
    candidate = query.strip()
    lowered = candidate.lower()
    for affix in _ARITHMETIC_AFFIXES:
        lowered = lowered.replace(affix, " ")
    # Re-apply the same removals positionally on the original-cased string by
    # rebuilding from the lowered version's structure: since the whitelist of
    # allowed characters (digits/operators/parentheses/whitespace) is
    # case-insensitive by nature (no letters survive), operating on the
    # lowered string is sufficient and keeps the logic simple.
    stripped = lowered.strip()
    if not stripped:
        return None
    if not _ARITHMETIC_CHARS_RE.match(stripped):
        return None
    if not _ARITHMETIC_OPERATOR_RE.search(stripped):
        return None
    # Normalize the caret (common "power" notation in natural language) to
    # Python's ``**`` before handing off to ``tools.calculate``.
    normalized = stripped.replace("^", "**")
    return normalized.strip()


class RuleRouter:
    """Deterministic, offline rule-based router (no LLM call).

    Default is always "rag" unless a rule matches with a clear, strong
    signal -- ambiguous queries stay on the existing RAG behavior so wiring
    this router in cannot regress Phase 2 answers.
    """

    def route(self, query: str) -> RouteDecision:
        lowered = query.strip().lower()

        # Rule 1 (checked first): datetime lookups.
        if any(keyword in lowered for keyword in _DATETIME_KEYWORDS):
            arguments: dict[str, Any] = {}
            if "giờ việt nam" in lowered or "việt nam" in lowered:
                arguments = {"timezone": "Asia/Ho_Chi_Minh"}
            elif "utc" in lowered:
                arguments = {"timezone": "UTC"}
            return RouteDecision(
                route="tool",
                reason="Câu hỏi chứa từ khoá ngày/giờ hiện tại; dùng tool current_datetime.",
                tool_name="current_datetime",
                arguments=arguments,
            )

        # Rule 2: arithmetic expressions.
        expression = _extract_arithmetic_expression(query)
        if expression is not None:
            return RouteDecision(
                route="tool",
                reason="Câu hỏi là một biểu thức số học; dùng tool calculate.",
                tool_name="calculate",
                arguments={"expression": expression},
            )

        return RouteDecision(route="rag", reason="Không khớp rule tool nào; dùng RAG.")


def build_router() -> Router:
    return RuleRouter()
