from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class AskRequest(BaseModel):
    message: str = Field(..., description="User input message")
    debug: bool = Field(default=False, description="Whether to include debug info in meta")
    use_colbert: bool = Field(default=True, description="Enable/disable ColBERT reranking per request")
    rag_enabled: bool = Field(default=True, description="Enable/disable RAG tools and retrieval")
    fusion_mode: str = Field(default="rrf", description="Fusion mode for hybrid retrieval: 'rrf' or 'lr'")
    session_id: Optional[str] = Field(default=None, description="Client-provided session id (optional)")
    max_steps: int = Field(default=6, description="Max tool-planning steps for the agent")
    agent_enabled: bool = Field(default=True, description="Allow planner + tools")
    web_enabled: bool = Field(default=False, description="Allow web tools (mcp.*)")


class AskResponse(BaseModel):
    user_question: str
    rewritten_query: str
    answer: str
    latency_ms: int
    usage: Dict[str, Any]
    hits: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None
