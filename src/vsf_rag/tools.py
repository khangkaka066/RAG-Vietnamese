from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: Callable[..., dict[str, Any]]


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

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]
