from __future__ import annotations

import pytest

from vsf_rag.tools import (
    ToolCallTrace,
    ToolError,
    ToolRegistry,
    build_default_registry,
    calculate,
    current_datetime,
)


# --- calculate ---------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+3*4", 14),
        ("(1+2)/4", 0.75),
        ("10 % 3", 1),
        ("-5 + 2", -3),
    ],
)
def test_calculate_evaluates_whitelisted_arithmetic(expression: str, expected: float) -> None:
    result = calculate(expression)
    assert result["expression"] == expression
    assert result["result"] == expected


def test_calculate_raises_tool_error_on_division_by_zero() -> None:
    with pytest.raises(ToolError):
        calculate("1/0")


def test_calculate_raises_tool_error_on_oversized_exponent() -> None:
    with pytest.raises(ToolError):
        calculate("9**999")


def test_calculate_raises_tool_error_on_nested_oversized_exponent() -> None:
    """Số mũ là biểu thức lồng nhau (không phải hằng số) vẫn phải bị chặn."""
    with pytest.raises(ToolError):
        calculate("9**(10**10)")


def test_calculate_raises_tool_error_on_complex_result() -> None:
    """Kết quả là số phức (vd căn bậc hai của số âm) phải bị từ chối rõ ràng."""
    with pytest.raises(ToolError):
        calculate("(-1)**0.5")


def test_calculate_raises_tool_error_on_non_finite_literal() -> None:
    """Literal tràn số (vd 1e309 -> inf) phải bị chặn, không được lọt qua ``result``."""
    with pytest.raises(ToolError):
        calculate("1e309")


def test_calculate_raises_tool_error_on_overflow_multiplication() -> None:
    """Phép nhân hai số float lớn tràn thành ``inf`` cũng phải bị chặn."""
    with pytest.raises(ToolError):
        calculate("1e308*1e308")


def test_calculate_raises_tool_error_on_huge_integer_result() -> None:
    """Số mũ hợp lệ (<=100) nhưng cơ số rất lớn vẫn có thể tạo số nguyên
    khổng lồ (~19.500 chữ số) mà không vi phạm guard số mũ. Kết quả này
    phải bị chặn để tránh crash khi nội suy vào chuỗi câu trả lời
    (CPython giới hạn 4300 chữ số khi chuyển int sang str)."""
    expression = "9" * 195 + "**100"
    assert len(expression) == 200
    with pytest.raises(ToolError):
        calculate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "abs(-1)",
        "x + 1",
        "'a' * 3",
        "[1, 2, 3]",
    ],
)
def test_calculate_rejects_non_arithmetic_syntax(expression: str) -> None:
    with pytest.raises(ToolError):
        calculate(expression)


def test_calculate_rejects_empty_expression() -> None:
    with pytest.raises(ToolError):
        calculate("   ")


# --- current_datetime ---------------------------------------------------


def test_current_datetime_defaults_to_vietnam_timezone_with_plus_seven_offset() -> None:
    result = current_datetime()
    assert result["timezone"] == "Asia/Ho_Chi_Minh"
    assert result["iso"].endswith("+07:00")
    # Must be parseable as an ISO timestamp.
    from datetime import datetime

    datetime.fromisoformat(result["iso"])
    assert result["weekday"] in {
        "Thứ Hai",
        "Thứ Ba",
        "Thứ Tư",
        "Thứ Năm",
        "Thứ Sáu",
        "Thứ Bảy",
        "Chủ Nhật",
    }


def test_current_datetime_accepts_other_iana_timezone() -> None:
    result = current_datetime("UTC")
    assert result["timezone"] == "UTC"
    assert result["iso"].endswith("+00:00")


def test_current_datetime_raises_tool_error_on_invalid_timezone() -> None:
    with pytest.raises(ToolError):
        current_datetime("Not/A_Real_Zone")


def test_current_datetime_raises_tool_error_on_none_timezone() -> None:
    with pytest.raises(ToolError):
        current_datetime(None)


def test_current_datetime_raises_tool_error_on_non_string_timezone() -> None:
    with pytest.raises(ToolError):
        current_datetime(123)


# --- ToolRegistry.call_with_trace ---------------------------------------


def test_call_with_trace_returns_success_trace() -> None:
    registry = build_default_registry()

    trace = registry.call_with_trace("calculate", {"expression": "2+2"})

    assert isinstance(trace, ToolCallTrace)
    assert trace.tool == "calculate"
    assert trace.error is None
    assert trace.result == {"expression": "2+2", "result": 4}
    assert trace.duration_ms >= 0
    assert trace.to_dict()["error"] is None


def test_call_with_trace_captures_unknown_tool_as_error() -> None:
    registry = build_default_registry()

    trace = registry.call_with_trace("does_not_exist", {})

    assert trace.result is None
    assert trace.error
    assert "does_not_exist" in trace.error


def test_call_with_trace_captures_tool_error() -> None:
    registry = build_default_registry()

    trace = registry.call_with_trace("calculate", {"expression": "1/0"})

    assert trace.result is None
    assert trace.error


def test_call_with_trace_captures_bad_arguments_as_error() -> None:
    registry = build_default_registry()

    trace = registry.call_with_trace("calculate", {"wrong_argument": "2+2"})

    assert trace.result is None
    assert trace.error


def test_call_with_trace_only_catches_caller_input_failures() -> None:
    """Genuine bugs inside a handler (not KeyError/ToolError/TypeError) must
    propagate -- ``call_with_trace`` only shields callers from bad
    tool-name/argument input, not from real implementation bugs."""
    registry = ToolRegistry()
    registry.register("boom", "always raises ZeroDivisionError", lambda: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        registry.call_with_trace("boom", {})


def test_build_default_registry_registers_calculate_and_current_datetime() -> None:
    registry = build_default_registry()
    names = {tool["name"] for tool in registry.describe()}
    assert {"calculate", "current_datetime"} <= names
