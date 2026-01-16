from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent.tools.base import Tool, ToolResult


@dataclass
class ToolRegistry:
    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name]

    def list_tools(self) -> List[Dict[str, str]]:
        out: List[Dict[str, Any]] = []
        for t in self.tools.values():
            out.append({
                "name": t.name,
                "description": t.description,
                "input_schema": getattr(t, "input_schema", {}),
            })
        return out

    @staticmethod
    def normalize_result(raw: Any) -> ToolResult:
        """Convert various tool return shapes to ToolResult."""
        if isinstance(raw, ToolResult):
            return raw
        if isinstance(raw, dict):
            return ToolResult(
                ok=bool(raw.get("ok", True)),
                content=raw.get("content") or raw.get("text"),
                artifacts=raw.get("artifacts") or {k: v for k, v in raw.items() if k not in {"ok", "content", "text", "debug", "error"}},
                debug=raw.get("debug"),
                error=raw.get("error"),
            )
        # Fallback: wrap as string content
        return ToolResult(ok=True, content=str(raw))
