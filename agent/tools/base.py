from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol


@dataclass
class ToolContext:
    """
    Per-request context passed to tools.
    Extend this structure as you add more runtime services (logger, cache, etc.).
    """
    session_id: str
    debug: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
        """
        Standard tool return shape for non-stream tools.

        Fields:
            - ok: success flag
            - content: main textual observation for the LLM
            - artifacts: structured payload (e.g., hits, rows, paths)
            - debug: optional debug info not necessarily shown to the user
            - error: error message when ok=False
        """
        ok: bool
        content: Optional[str] = None
        artifacts: Optional[Dict[str, Any]] = None
        debug: Optional[Dict[str, Any]] = None
        error: Optional[str] = None


class Tool(Protocol):
    """
    Minimal tool protocol.
    Tools may implement either run() or run_stream() depending on their nature.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]

    def run(self, ctx: ToolContext, **kwargs) -> Dict[str, Any]:
        ...

    def run_stream(self, ctx: ToolContext, **kwargs) -> Iterable[Dict[str, Any]]:
        ...
