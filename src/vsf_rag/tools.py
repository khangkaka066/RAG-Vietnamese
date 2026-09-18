from __future__ import annotations

import ast
import math
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ToolError(RuntimeError):
    """Raised for any "bad input / could not compute" failure inside a tool.

    Every tool function in this module raises exactly this type on failure
    so ``ToolRegistry`` (and its callers) only need to catch one exception
    to distinguish "tool ran but rejected the input" from an unexpected bug.
    """


_MAX_EXPRESSION_LENGTH = 200
_MAX_POWER_EXPONENT = 100
# Any int/float result (or intermediate sub-result) whose absolute value
# exceeds this is rejected. Chosen generously below CPython's default 4300
# digit limit for int<->str conversion (str(10**300) is 301 digits) so that
# a rejected result can always still be safely formatted into an error
# message, while comfortably covering every legitimate calculator use case.
_MAX_RESULT_ABS = 1e300

# Whitelisted binary/unary operators for ``calculate``. Anything not listed
# here (bitwise ops, comparisons, boolean ops, matmul, ...) is rejected by
# ``_eval_node`` regardless of what ``ast`` parses, because the walker only
# ever dispatches through this mapping.
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_WEEKDAYS_VI = [
    "Thứ Hai",
    "Thứ Ba",
    "Thứ Tư",
    "Thứ Năm",
    "Thứ Sáu",
    "Thứ Bảy",
    "Chủ Nhật",
]


def _check_result_bounds(result: int | float) -> None:
    """Reject non-finite floats and absurdly large int/float results.

    Called after *every* ``BinOp`` evaluation (not just at the top level of
    ``calculate``) so an oversized/non-finite intermediate value is rejected
    as soon as it appears, before it can be fed into further arithmetic or
    ever reach a caller (e.g. ``answering.py`` interpolating it into a
    string, which would otherwise crash on CPython's int->str digit limit).
    """
    if isinstance(result, float) and not math.isfinite(result):
        raise ToolError("Kết quả không phải số hữu hạn, không được hỗ trợ.")
    if abs(result) > _MAX_RESULT_ABS:
        raise ToolError("Kết quả quá lớn để tính toán.")


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(
                f"Biểu thức chỉ được chứa số, không được chứa: {node.value!r}"
            )
        _check_result_bounds(node.value)
        return node.value

    if isinstance(node, ast.BinOp):
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ToolError(f"Toán tử không được hỗ trợ: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER_EXPONENT:
            raise ToolError(
                f"Số mũ quá lớn (tối đa {_MAX_POWER_EXPONENT}), tránh tính toán quá tải."
            )
        try:
            result = op_fn(left, right)
        except ZeroDivisionError as exc:
            raise ToolError("Không thể chia cho 0.") from exc
        except OverflowError as exc:
            raise ToolError("Kết quả quá lớn để tính toán.") from exc
        if isinstance(result, complex):
            raise ToolError("Biểu thức cho kết quả là số phức, không được hỗ trợ.")
        _check_result_bounds(result)
        return result

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ToolError(f"Toán tử một ngôi không được hỗ trợ: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))

    raise ToolError(f"Biểu thức chứa cú pháp không được phép: {type(node).__name__}")


def calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate a whitelisted arithmetic expression.

    Only numeric constants, ``+ - * / // % **`` and unary +/- are allowed.
    No ``eval``/``literal_eval`` is ever called on untrusted input: the
    expression is parsed with ``ast.parse`` and then walked/evaluated
    recursively node-by-node against an explicit whitelist (see
    ``_eval_node``), rejecting anything else (names, calls, attributes,
    subscripts, comprehensions, strings, ...).
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("Biểu thức không được để trống.")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Biểu thức quá dài (tối đa {_MAX_EXPRESSION_LENGTH} ký tự).")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Biểu thức không hợp lệ: {expression!r}") from exc

    result = _eval_node(tree)
    return {"expression": expression.strip(), "result": result}


def current_datetime(timezone: str = "Asia/Ho_Chi_Minh") -> dict[str, Any]:
    """Look up the current date/time in the given IANA timezone (stdlib ``zoneinfo``).

    ``timezone`` must be a ``str``; any other type (``None``, ``int``, ...)
    is rejected as a ``ToolError`` rather than being passed to ``ZoneInfo``
    (which would raise a raw ``TypeError``), so this tool always fails with
    the same exception type regardless of what bad input it is given.
    """
    if not isinstance(timezone, str):
        raise ToolError(f"Múi giờ không hợp lệ: {timezone!r}")
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(f"Múi giờ không hợp lệ: {timezone!r}") from exc

    now = datetime.now(zone)
    return {
        "timezone": timezone,
        "iso": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.time().isoformat(timespec="seconds"),
        "weekday": _WEEKDAYS_VI[now.weekday()],
    }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ToolCallTrace:
    """Auditable record of a single tool invocation, success or failure."""

    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class ToolRegistry:
    """Explicit registry for validated, auditable tool calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, handler: Callable[..., dict[str, Any]]) -> None:
        if not name or name in self._tools:
            raise ValueError("Tool name must be non-empty and unique")
        self._tools[name] = Tool(name, description, handler)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(**arguments)

    def call_with_trace(self, name: str, arguments: dict[str, Any]) -> ToolCallTrace:
        """Call a tool, capturing success/failure into a ``ToolCallTrace`` instead of raising.

        Catches ``KeyError`` (unknown tool name), ``ToolError`` (tool-reported
        bad input), and ``TypeError`` (wrong/missing/extra keyword arguments
        in ``arguments`` not matching the handler's signature, e.g. a caller
        passing ``{"tz": ...}`` instead of ``{"timezone": ...}``) -- these are
        the three ways a caller-controlled ``name``/``arguments`` pair can
        fail without it being a genuine bug in the tool implementation
        itself. Individual tools are expected to validate the *values* of
        their own arguments and raise ``ToolError`` themselves (see
        ``current_datetime``'s explicit ``isinstance`` check) rather than
        relying on this ``TypeError`` catch, which only guards against
        Python raising it for mismatched argument *names/arity* at the call
        boundary. Any other exception is a real bug and is allowed to
        propagate.
        """
        start = time.perf_counter()
        try:
            result = self.call(name, arguments)
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            return ToolCallTrace(
                tool=name,
                arguments=arguments,
                result=result,
                error=None,
                duration_ms=duration_ms,
            )
        except (KeyError, ToolError, TypeError) as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            return ToolCallTrace(
                tool=name,
                arguments=arguments,
                result=None,
                error=str(exc),
                duration_ms=duration_ms,
            )

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]


def build_default_registry(extra: dict[str, tuple[str, Callable[..., dict[str, Any]]]] | None = None) -> ToolRegistry:
    """Build a ``ToolRegistry`` pre-populated with the built-in offline tools.

    ``extra`` optionally maps additional tool names to ``(description, handler)``
    pairs registered on top of the defaults (used by ``api.py`` to add
    ``list_sources`` without duplicating registry construction).
    """
    registry = ToolRegistry()
    registry.register(
        "calculate",
        "Tính một biểu thức số học an toàn (vd: '2 + 3 * 4').",
        calculate,
    )
    registry.register(
        "current_datetime",
        "Tra cứu ngày giờ hiện tại theo múi giờ (mặc định Asia/Ho_Chi_Minh).",
        current_datetime,
    )
    for name, (description, handler) in (extra or {}).items():
        registry.register(name, description, handler)
    return registry
