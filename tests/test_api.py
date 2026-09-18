from __future__ import annotations

import vsf_rag.api as api_module
from fastapi.testclient import TestClient

from vsf_rag.api import app, get_engine, tools as api_tools


client = TestClient(app)


def test_health_reports_generator_matching_engine_state() -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert "generator" in body
    assert body["generator"] in {"llm", "extractive"}
    # The reported value must reflect whether the running engine actually
    # has a working LLM provider wired up (see .bangiao/ke-hoach.md bước 17).
    # ``get_engine()`` is only safe to call *inside* a test (after the
    # autouse env-cleanup fixture has run) -- see
    # test_engine_is_not_built_at_import_time below for why.
    engine = get_engine()
    expected = "llm" if engine.llm is not None else "extractive"
    assert body["generator"] == expected

    # Existing fields must still be present (no regression).
    assert body["status"] == "ok"
    assert body["documents"] > 0
    assert body["version"] == app.version
    assert body["retriever"] == type(engine.retriever).__name__


def test_engine_is_not_built_at_import_time(monkeypatch) -> None:
    """Regression test for Codex review round 2, finding #4.

    ``vsf_rag.api`` used to call ``build_answer_engine()`` -- which reads
    ``OPENROUTER_API_KEY``/``OPENROUTER_MODEL`` from the environment -- at
    *module import time*. Since pytest imports every test module during
    collection (before any test's fixtures run), that meant a real
    ``OPENROUTER_API_KEY`` present in the developer/CI environment could
    leak into a live ``OpenRouterProvider`` before the autouse
    ``_clean_openrouter_env`` fixture in ``conftest.py`` ever got a chance
    to scrub it, breaking test isolation.

    This test proves the engine is instead built lazily on first use, by
    resetting the ``lru_cache`` and confirming that setting
    ``OPENROUTER_API_KEY`` *after* import (simulating "the module was
    already imported before this fixture ran") is still correctly picked
    up the first time ``get_engine()`` actually runs.
    """
    api_module.get_engine.cache_clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake-key-does-not-exist")

    engine = api_module.get_engine()

    assert engine.llm is not None
    api_module.get_engine.cache_clear()


def test_query_arithmetic_routes_to_calculate_tool() -> None:
    response = client.post("/query", json={"query": "tính 2+3*4"})
    assert response.status_code == 200

    body = response.json()
    assert body["route"] == "tool"
    assert body["status"] == "OK"
    assert body["tool_trace"]
    assert body["tool_trace"][0]["tool"] == "calculate"
    assert body["tool_trace"][0]["error"] is None


def test_query_datetime_routes_to_current_datetime_tool() -> None:
    response = client.post("/query", json={"query": "Hôm nay là ngày mấy?"})
    assert response.status_code == 200

    body = response.json()
    assert body["route"] == "tool"
    assert body["tool_trace"][0]["tool"] == "current_datetime"


def test_query_rag_query_has_empty_tool_trace() -> None:
    response = client.post(
        "/query", json={"query": "RAG cần trích dẫn nguồn như thế nào?"}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["route"] == "rag"
    assert body["route_reason"]
    assert body["tool_trace"] == []


def test_list_tools_returns_registry_description() -> None:
    """``GET /tools`` should expose exactly what ``ToolRegistry.describe()``
    reports for the module-level registry used by ``/tools/call``, so the
    two endpoints never drift apart."""
    response = client.get("/tools")
    assert response.status_code == 200
    body = response.json()
    assert body == {"tools": api_tools.describe()}
    names = {tool["name"] for tool in body["tools"]}
    assert "calculate" in names
    assert "current_datetime" in names


def test_tools_call_unknown_tool_returns_404() -> None:
    response = client.post("/tools/call", json={"name": "does_not_exist", "arguments": {}})
    assert response.status_code == 404


def test_tools_call_bad_arguments_returns_400() -> None:
    response = client.post(
        "/tools/call", json={"name": "calculate", "arguments": {"expression": "1/0"}}
    )
    assert response.status_code == 400


def test_tools_call_calculate_succeeds() -> None:
    response = client.post(
        "/tools/call", json={"name": "calculate", "arguments": {"expression": "2+2"}}
    )
    assert response.status_code == 200
    assert response.json()["result"]["result"] == 4


def test_evaluate_returns_metrics() -> None:
    response = client.post("/evaluate", json={"top_k": 3})
    assert response.status_code == 200

    body = response.json()
    assert body["cases"] == 20
    for field in (
        "retrieval_hit_rate",
        "retrieval_mrr_at_k",
        "citation_coverage",
        "faithfulness",
        "answer_relevancy",
        "latency_ms",
        "routes",
    ):
        assert field in body


def test_evaluate_omits_details_by_default() -> None:
    response = client.post("/evaluate", json={"top_k": 3})
    assert response.status_code == 200
    assert "details" not in response.json()


def test_evaluate_include_details() -> None:
    response = client.post("/evaluate", json={"top_k": 3, "include_details": True})
    assert response.status_code == 200

    body = response.json()
    assert "details" in body
    assert len(body["details"]) == body["cases"]


def test_evaluate_with_inline_cases() -> None:
    response = client.post(
        "/evaluate",
        json={
            "top_k": 3,
            "cases": [
                {
                    "query": "RAG cần trích dẫn nguồn như thế nào?",
                    "expected_doc_ids": [],
                    "required_terms": [],
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["cases"] == 1


def test_evaluate_rejects_empty_cases_list() -> None:
    response = client.post("/evaluate", json={"top_k": 3, "cases": []})
    assert response.status_code == 422


def test_evaluate_rejects_cases_over_limit() -> None:
    """``cases`` is capped at 50 (Codex review round 3, finding #1): an
    unbounded list is a cost/DoS risk once ``OPENROUTER_API_KEY`` is set,
    since each case triggers one LLM call."""
    case = {"query": "RAG cần trích dẫn nguồn như thế nào?"}
    response = client.post("/evaluate", json={"top_k": 3, "cases": [case] * 51})
    assert response.status_code == 422


def test_evaluate_empty_eval_set_returns_400(monkeypatch) -> None:
    """When ``cases`` is omitted and the checked-in eval set is empty,
    ``evaluation.evaluate()`` raises ``ValueError`` -- the endpoint should
    map that to 400 (a runtime/data condition), not 422 (a request-shape
    validation failure, which is what an over-limit/empty *inline* ``cases``
    list already gets from pydantic before the handler even runs)."""
    monkeypatch.setattr(api_module, "get_eval_cases", lambda: ())
    response = client.post("/evaluate", json={"top_k": 3})
    assert response.status_code == 400


def test_openapi_has_query_example() -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    query_request_schema = schema["components"]["schemas"]["QueryRequest"]
    assert "examples" in query_request_schema or "example" in query_request_schema


def test_openapi_evaluate_response_schema_matches_evaluate_output() -> None:
    """``/evaluate`` must advertise a real response schema/example -- not just
    the generic ``dict`` FastAPI infers when no ``response_model`` is set
    (Codex review round 3, finding #2)."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    evaluate_op = schema["paths"]["/evaluate"]["post"]
    response_schema_ref = evaluate_op["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema_ref  # a $ref (or resolved schema), not empty

    request_schema = schema["components"]["schemas"]["EvaluateRequest"]
    assert "examples" in request_schema or "example" in request_schema

    evaluate_response_schema = schema["components"]["schemas"]["EvaluateResponse"]
    assert "examples" in evaluate_response_schema or "example" in evaluate_response_schema
    for field in (
        "retrieval_hit_rate",
        "retrieval_mrr_at_k",
        "citation_coverage",
        "faithfulness",
        "answer_relevancy",
        "latency_ms",
        "routes",
        "details",
    ):
        assert field in evaluate_response_schema["properties"]

    latency_stats_schema = schema["components"]["schemas"]["LatencyStats"]
    for field in (
        "retrieval_mean",
        "retrieval_p95",
        "generation_mean",
        "generation_p95",
        "total_mean",
        "total_p95",
    ):
        assert field in latency_stats_schema["properties"]


def test_evaluate_response_matches_real_evaluate_output() -> None:
    """``EvaluateResponse`` must not drop or misname any key that
    ``evaluation.evaluate()`` actually returns (Codex review round 3,
    finding #2)."""
    response = client.post("/evaluate", json={"top_k": 3, "include_details": True})
    assert response.status_code == 200
    body = response.json()

    from vsf_rag.evaluation import evaluate as run_evaluate

    real_report = run_evaluate(get_engine(), list(api_module.get_eval_cases()), top_k=3)
    assert set(real_report.keys()) == set(body.keys())
    assert set(real_report["latency_ms"].keys()) == set(body["latency_ms"].keys())


def test_query_response_shape_matches_answer_dataclass() -> None:
    """The ``QueryResponse`` model used for ``/query`` docs must not drop any
    field ``Answer.to_dict()`` produces (see .bangiao/ke-hoach.md)."""
    response = client.post("/query", json={"query": "tính 2+3*4"})
    assert response.status_code == 200

    body = response.json()
    from vsf_rag.answering import Answer

    expected_fields = {field for field in Answer.__dataclass_fields__}
    assert expected_fields == set(body.keys())
