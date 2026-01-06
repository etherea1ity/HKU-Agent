from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

from rag.config import RagConfig
from rag.pipeline.answer import answer as rag_answer

from agent.types import AgentState, Message, TraceEvent, TurnResult


@dataclass
class RagAgent:
    """
    Minimal agent that uses your RAG pipeline.
    No LLM call yet: we return prompt/context for inspection.

    Later:
      - plug in generate() to call an LLM
      - add tool use / MCP
      - add ColBERT rerank hook in retrieval
    """
    cfg: RagConfig

    def run(self, state: AgentState, user_text: str, run_id: Optional[str] = None) -> TurnResult:
        if run_id is None:
            run_id = f"run_{int(time.time()*1000)}"

        events = []
        events.append(TraceEvent(name="turn_start", attrs={"conversation_id": state.conversation_id}))

        # 1) Update conversation history
        state.messages.append(Message(role="user", content=user_text))
        events.append(TraceEvent(name="history_update", attrs={"messages": len(state.messages)}))

        # 2) RAG pipeline
        t0 = time.time()
        out = rag_answer(self.cfg, user_text)
        t1 = time.time()

        events.append(TraceEvent(
            name="rag_pipeline_done",
            attrs={
                "latency_ms": int((t1 - t0) * 1000),
                "query": out["query"],
                "num_hits": len(out["hits"]),
                "context_chars": len(out["context"]),
                "prompt_chars": len(out["prompt"]),
            }
        ))

        # 3) (Future) LLM generation would go here
        answer_text = None

        # 4) Add assistant message placeholder (optional)
        # If you later generate, replace None with real answer text.
        state.messages.append(Message(role="assistant", content=answer_text or "[no-llm-output-yet]"))

        events.append(TraceEvent(name="turn_end"))

        return TurnResult(
            run_id=run_id,
            user_text=user_text,
            query=out["query"],
            context=out["context"],
            prompt=out["prompt"],
            answer_text=answer_text,
            events=events,
        )
