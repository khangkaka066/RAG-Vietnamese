from __future__ import annotations

import pytest

from vsf_rag.router import RouteDecision, RuleRouter, build_router


@pytest.fixture()
def router() -> RuleRouter:
    return RuleRouter()


@pytest.mark.parametrize(
    "query",
    [
        "2+3*4",
        "tính 2+3*4",
        "(1+2)/4",
        "10 % 3",
        "-5 + 2 bằng bao nhiêu",
    ],
)
def test_router_routes_arithmetic_queries_to_calculate(router: RuleRouter, query: str) -> None:
    decision = router.route(query)
    assert decision.route == "tool"
    assert decision.tool_name == "calculate"
    assert "expression" in decision.arguments


@pytest.mark.parametrize(
    "query",
    [
        "Hôm nay là ngày mấy?",
        "Bây giờ là mấy giờ ở Việt Nam?",
        "current time in UTC",
    ],
)
def test_router_routes_datetime_queries_to_current_datetime(router: RuleRouter, query: str) -> None:
    decision = router.route(query)
    assert decision.route == "tool"
    assert decision.tool_name == "current_datetime"


@pytest.mark.parametrize(
    "query",
    [
        "RAG cần trích dẫn nguồn như thế nào?",
        "điều 5",
        "2024",
        "Chunking và overlap giữa các đoạn văn bản dùng để làm gì trong RAG?",
    ],
)
def test_router_defaults_to_rag_for_non_tool_queries(router: RuleRouter, query: str) -> None:
    decision = router.route(query)
    assert decision.route == "rag"
    assert decision.tool_name is None


def test_route_decision_reason_is_non_empty() -> None:
    router = RuleRouter()
    decision = router.route("2+2")
    assert decision.reason


def test_build_router_returns_rule_router() -> None:
    router = build_router()
    assert isinstance(router, RuleRouter)


def test_route_decision_defaults() -> None:
    decision = RouteDecision(route="rag", reason="test")
    assert decision.tool_name is None
    assert decision.arguments == {}
