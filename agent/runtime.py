from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, List

from agent.planner import LLMPlanner
from agent.memory import SessionMemory
from agent.tools.base import ToolContext
from agent.tools.registry import ToolRegistry
from core.llm_client import LLMClient


@dataclass
class AgentRuntime:
    registry: ToolRegistry
    planner: LLMPlanner
    llm: LLMClient
    sessions: Dict[str, SessionMemory] = field(default_factory=dict)

    def _get_session_id(self, session_id: Optional[str]) -> str:
        return session_id or uuid.uuid4().hex

    def _get_memory(self, sid: str) -> SessionMemory:
        if sid not in self.sessions:
            self.sessions[sid] = SessionMemory(session_id=sid)
        return self.sessions[sid]

    def run_stream(
        self,
        user_text: str,
        *,
        session_id: Optional[str] = None,
        debug: bool = False,
        use_colbert: bool = True,
        rag_enabled: bool = True,
        agent_enabled: bool = True,
        web_enabled: bool = False,
        fusion_mode: str = "rrf",
        max_steps: int = 6,
    ) -> Iterable[Dict[str, Any]]:
        """
        Multi-step tool-using loop.
        Emits events compatible with your existing SSE packing:
          - tool_start / tool_end
          - meta / delta / usage / done / error (from rag.qa or rag.answer if you later stream it)
        """
        sid = self._get_session_id(session_id)
        memory = self._get_memory(sid)

        ctx = ToolContext(session_id=sid, debug=debug, extra={"fusion_mode": fusion_mode})

        # Prepare tool list based on feature flags.
        tools_for_planner = self.registry.list_tools()
        if not rag_enabled:
            tools_for_planner = [t for t in tools_for_planner if not t["name"].startswith("rag.")]
        if not web_enabled:
            tools_for_planner = [t for t in tools_for_planner if not t["name"].startswith("mcp.")]

        # Provide a lightweight agent-level meta event.
        yield {
            "type": "agent_meta",
            "session_id": sid,
            "max_steps": max_steps,
            "tools": tools_for_planner,
        }

        # If agent is disabled, short-circuit to a direct RAG answer (if allowed).
        if not agent_enabled:
            if not rag_enabled:
                # Direct single-shot LLM answer as a fallback
                t0 = time.time()
                try:
                    text, usage = self.llm.chat([
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_text},
                    ])
                except Exception as e:
                    yield {"type": "error", "message": f"LLM fallback failed: {e}"}
                    yield {"type": "done", "answer": "", "latency_ms": 0, "usage": {}}
                    return

                latency_ms = int((time.time() - t0) * 1000)
                if text:
                    yield {"type": "delta", "text": text}
                yield {
                    "type": "done",
                    "answer": text or "",
                    "latency_ms": latency_ms,
                    "usage": {"llm": usage} if usage else {},
                }
                return

            try:
                rag_tool = self.registry.get("rag.qa")
            except Exception:
                yield {"type": "error", "message": "RAG tool not available."}
                yield {"type": "done", "answer": "", "latency_ms": 0, "usage": {}}
                return

            t0 = time.time()
            final_text: List[str] = []
            for ev in rag_tool.run_stream(ctx, question=user_text, use_colbert=use_colbert, include_debug=debug):
                if ev.get("type") == "delta":
                    final_text.append(ev.get("text", ""))
                yield ev
            answer_text = "".join(final_text)
            latency_ms = int((time.time() - t0) * 1000)
            yield {
                "type": "done",
                "answer": answer_text,
                "latency_ms": latency_ms,
                "usage": {},
            }
            return

        t0 = time.time()

        for step in range(1, max_steps + 1):
            plan, plan_usage = self.planner.plan_next(user_text, tools_for_planner, memory.to_planner_state())

            yield {
                "type": "plan_update",
                "step": step,
                "tool": plan.tool,
                "args": plan.args,
                "thought": plan.thought,
            }

            if plan.stop and plan.tool == "final":
                latency_ms = int((time.time() - t0) * 1000)
                memory.record(step, tool="final", args={}, result=self.registry.normalize_result({"ok": True, "content": plan.final_answer}), thought=plan.thought, latency_ms=latency_ms)
                # Emit a single delta so front-end can show immediate text for non-stream replies.
                if plan.final_answer:
                    yield {"type": "delta", "text": plan.final_answer}
                yield {
                    "type": "done",
                    "answer": plan.final_answer or "",
                    "latency_ms": latency_ms,
                    "usage": {"planner": plan_usage},
                }
                return

            # Tool execution
            try:
                tool = self.registry.get(plan.tool)
            except Exception as e:
                yield {"type": "error", "message": f"Unknown tool from planner: {plan.tool} ({e})"}
                return

            # Special-case: rag.qa streams meta/delta/usage/done already.
            if plan.tool == "rag.qa":
                q = plan.args.get("question") or user_text
                final_text = []
                for ev in tool.run_stream(ctx, question=q, use_colbert=use_colbert, include_debug=debug):
                    if ev.get("type") == "delta":
                        final_text.append(ev.get("text", ""))
                    yield ev
                answer_text = "".join(final_text)
                norm_final = self.registry.normalize_result({"ok": True, "content": answer_text})
                memory.record(step, tool=plan.tool, args={"question": q}, result=norm_final, thought=plan.thought)
                yield {"type": "tool_end", "step": step, "tool": plan.tool}
                latency_ms = int((time.time() - t0) * 1000)
                yield {
                    "type": "done",
                    "answer": answer_text,
                    "latency_ms": latency_ms,
                    "usage": {"planner": plan_usage},
                }
                return

            # Non-stream tools: run once, store to memory, continue.
            try:
                raw_result = tool.run(ctx, **plan.args)
            except Exception as e:
                yield {"type": "error", "message": f"Tool execution failed: {plan.tool} ({e})"}
                return

            norm = self.registry.normalize_result(raw_result)
            memory.record(step, tool=plan.tool, args=plan.args, result=norm, thought=plan.thought)

            yield {
                "type": "tool_end",
                "step": step,
                "tool": plan.tool,
                "result": {
                    "ok": norm.ok,
                    "content": norm.content,
                    "artifacts": norm.artifacts,
                    "error": norm.error,
                },
            }

            # Heuristic: if planner called rag.answer and succeeded, stop.
            if plan.tool == "rag.answer" and norm.ok and norm.content:
                latency_ms = int((time.time() - t0) * 1000)
                yield {
                    "type": "done",
                    "answer": norm.content,
                    "latency_ms": latency_ms,
                    "usage": {"planner": plan_usage},
                }
                return

        # Soft stop: return best-effort summary instead of raising.
        yield {
            "type": "done",
            "answer": "Reached max steps without final answer. See trace for details.",
            "latency_ms": int((time.time() - t0) * 1000),
            "usage": {},
        }
