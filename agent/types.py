from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import time
import uuid


Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: Role
    content: str
    ts: float = field(default_factory=lambda: time.time())


@dataclass
class TraceEvent:
    """
    One structured event for observability.
    This is similar to your obs/json logger events, but kept in-memory first.
    """
    name: str
    ts: float = field(default_factory=lambda: time.time())
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    """
    The output of one agent turn.
    For now, we don't call an LLM, so 'answer_text' is None.
    We return the RAG prompt + retrieval evidence so you can inspect/debug.
    """
    run_id: str
    user_text: str
    query: str
    context: str
    prompt: str
    answer_text: Optional[str] = None
    events: List[TraceEvent] = field(default_factory=list)


@dataclass
class AgentState:
    """
    Minimal multi-turn state.
    Later you can add:
      - memory summaries
      - tool results
      - user profile, preferences, etc.
    """
    conversation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: List[Message] = field(default_factory=list)
