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
