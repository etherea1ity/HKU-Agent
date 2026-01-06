from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

class AskRequest(BaseModel):
    message: str = Field(..., description="User input message")
    debug: bool = Field(default=False, description="Whether to return internal fields")

class AskResponse(BaseModel):
    user_question: str
    rewritten_query: str
    answer: str
    latency_ms: int
    usage: Dict[str, Any]
    hits: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None
