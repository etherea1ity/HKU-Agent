from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.tools.base import ToolContext, ToolResult
from rag.config import RagConfig
from rag.pipeline.retrieve import retrieve
from rag.pipeline.context import build_context, expand_hits
from core.llm_client import LLMClient
from core.prompts import make_answer_messages
from agent.flow_rag import RagFlow


def _safe_read_text(path: Path) -> str:
    """
    Read text robustly on Windows and mixed encodings.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


@dataclass
class RagSearchTool:
    """
    Retrieve relevant chunks but do not generate an answer.
    """
    cfg: RagConfig
    name: str = "rag.search"
    description: str = "Search local corpus and return ranked evidence chunks."
    input_schema: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "use_colbert": {"type": "boolean"},
                "fusion_mode": {"type": "string"},
            },
            "required": ["query"],
        }

    def run(
        self,
        ctx: ToolContext,
        *,
        query: str,
        use_colbert: bool = True,
        fusion_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        from dataclasses import replace

        fm = fusion_mode or (ctx.extra.get("fusion_mode") if ctx.extra else None)
        cfg = self.cfg
        if fm:
            cfg = replace(self.cfg, fusion_mode=str(fm))

        hits = retrieve(cfg, query, use_colbert=use_colbert)
        return ToolResult(
            ok=True,
            content=f"retrieved {len(hits)} hits",
            artifacts={"hits": hits},
        )


@dataclass
class RagOpenTool:
    """
    Open original source text by (source_path, line_start, line_end).
    This is the key tool for citations and "show me the original paragraph".
    """
    project_root: Path
    name: str = "rag.open"
    description: str = "Open original source file content by line range."
    input_schema: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "source_path": {"type": "string"},
                "line_start": {"type": "integer"},
                "line_end": {"type": "integer"},
                "max_chars": {"type": "integer"},
            },
            "required": ["source_path", "line_start", "line_end"],
        }

    def run(
        self,
        ctx: ToolContext,
        *,
        source_path: str,
        line_start: int,
        line_end: int,
        max_chars: int = 4000,
    ) -> ToolResult:
        # Normalize Windows backslashes and resolve relative to repo root.
        rel = Path(source_path.replace("\\", "/"))
        abs_path = (self.project_root / rel).resolve()

        if not abs_path.exists():
            return ToolResult(ok=False, error=f"File not found: {abs_path}")

        text = _safe_read_text(abs_path)
        lines = text.splitlines()

        # Input line indices are 1-based in your metadata; clamp safely.
        s = max(1, int(line_start))
        e = max(s, int(line_end))
        s0 = s - 1
        e0 = min(len(lines), e)

        span = "\n".join(lines[s0:e0]).strip()
        if len(span) > max_chars:
            span = span[:max_chars] + "\n...[truncated]"

        return ToolResult(
            ok=True,
            content=span,
            artifacts={
                "source_path": str(rel),
                "line_start": s,
                "line_end": e0,
            },
        )


@dataclass
class RagAnswerTool:
    """
    Generate an answer using provided evidence text (no retrieval inside).
    This enables multi-step: search -> open -> answer.
    """
    llm: LLMClient
    name: str = "rag.answer"
    description: str = "Answer using provided evidence text (no retrieval)."
    input_schema: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["question", "evidence"],
        }

    def run(self, ctx: ToolContext, *, question: str, evidence: str) -> Dict[str, Any]:
        messages = make_answer_messages(question, evidence)
        text, usage = self.llm.chat(messages)
        return ToolResult(
            ok=True,
            content=text,
            artifacts={"usage": usage},
        )


@dataclass
class RagQATool:
    """
    End-to-end RAG tool that streams answer tokens.
    This is your existing RagFlow wrapped as a tool.
    """
    flow: RagFlow
    name: str = "rag.qa"
    description: str = "End-to-end RAG: rewrite + retrieve + context + answer (stream)."
    input_schema: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "use_colbert": {"type": "boolean"},
                "include_debug": {"type": "boolean"},
                "fusion_mode": {"type": "string"},
            },
            "required": ["question"],
        }

    def run_stream(
        self,
        ctx: ToolContext,
        *,
        question: str,
        use_colbert: bool = True,
        include_debug: bool = False,
        fusion_mode: Optional[str] = None,
    ) -> Iterable[Dict[str, Any]]:
        fm = fusion_mode or (ctx.extra.get("fusion_mode") if ctx.extra else None)
        # Forward existing RagFlow stream events.
        yield from self.flow.run_stream(
            question,
            include_debug=include_debug,
            use_colbert=use_colbert,
            fusion_mode=fm,
        )
