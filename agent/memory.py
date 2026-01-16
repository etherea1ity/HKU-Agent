from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.tools.base import ToolResult


@dataclass
class StepRecord:
    step: int
    tool: str
    args: Dict[str, Any]
    ok: bool
    content: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None
    thought: Optional[str] = None
    latency_ms: Optional[int] = None

    def to_planner_view(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "content": self.content,
            "artifacts": self.artifacts,
            "error": self.error,
        }


@dataclass
class SessionMemory:
    session_id: str
    created_at: float = field(default_factory=time.time)
    steps: List[StepRecord] = field(default_factory=list)
    summary: Optional[str] = None  # placeholder for future summarization

    def record(self, step: int, *, tool: str, args: Dict[str, Any], result: ToolResult, thought: Optional[str], latency_ms: Optional[int] = None) -> None:
        rec = StepRecord(
            step=step,
            tool=tool,
            args=args,
            ok=result.ok,
            content=result.content,
            artifacts=result.artifacts,
            error=result.error,
            debug=result.debug,
            thought=thought,
            latency_ms=latency_ms,
        )
        self.steps.append(rec)

    def to_planner_state(self) -> Dict[str, Any]:
        """Lightweight view for planner prompt; avoid dumping huge artifacts."""
        return {
            "steps": [s.to_planner_view() for s in self.steps][-12:],  # keep recent
            "summary": self.summary,
        }

