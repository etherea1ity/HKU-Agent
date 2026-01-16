from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.llm_client import LLMClient
from core.prompts import AGENT_IDENTITY


def _load_avatar_system_patch() -> Optional[str]:
    """Load planner system prompt patch from AVATAR memory bank.

    Enable via:
      - env AVATAR_ENABLED=1
      - optional env AVATAR_POLICY_PATH=<path to memory bank record>
    """

    if os.getenv("AVATAR_ENABLED", "0") not in ("1", "true", "True"):
        return None

    try:
        from avatar.memory_bank import resolve_best_policy_path, load_policy, extract_system_patch
    except Exception:
        return None

    p = os.getenv("AVATAR_POLICY_PATH")
    if not p:
        p = resolve_best_policy_path()
    if not p:
        return None

    try:
        rec = load_policy(p)
        return extract_system_patch(rec)
    except Exception:
        return None


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON robustly. Also supports fenced blocks if the model returns them.
    """
    s = text.strip()

    # Remove common markdown fences.
    if s.startswith("```"):
        s = s.strip("`")
        # In case of ```json\n...\n```
        s = s.replace("json\n", "", 1).strip()

    # Attempt direct parse.
    try:
        return json.loads(s)
    except Exception:
        pass

    # Attempt to extract the first JSON object.
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(s[l : r + 1])
        except Exception:
            return None
    return None


def _is_small_talk(text: str) -> bool:
    s = text.lower()
    return any(k in s for k in ["hi", "hello", "hey", "who are you", "your name", "what are you", "introduce yourself"])


def _default_identity_answer() -> str:
    return "I am HKU-Agent, a helpful assistant for HKU Computer Science postgraduate courses."


@dataclass
class PlanStep:
    tool: str
    args: Dict[str, Any]
    stop: bool = False
    final_answer: Optional[str] = None
    thought: Optional[str] = None


@dataclass
class LLMPlanner:
    llm: LLMClient

    def _build_messages(
        self,
        user_question: str,
        tools: List[Dict[str, Any]],
        memory: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        The planner must output JSON only.
        """
        tool_lines = []
        for t in tools:
            tool_lines.append(f"- {t['name']}: {t['description']}\n  schema: {json.dumps(t['input_schema'], ensure_ascii=False)}")

        memory_brief = json.dumps(memory, ensure_ascii=False)[:4000]

        avatar_patch = _load_avatar_system_patch()

        system = (
            f"{AGENT_IDENTITY}\n"
            "You are an agent planner.\n"
            "Pick ONE action each turn and output ONLY JSON that matches the schema.\n"
            "Actions must be from the provided tool list or 'final'.\n"
            "Use evidence already in memory before searching again.\n"
            "Use rag.* when answering local course facts; use mcp.* (web search / web parser) for external or fresh internet information.\n"
            "Only call retrieval tools (rag.* or mcp.*) when the user asks for factual details that need evidence.\n"
            "If the user asks for external, real-time or non-course info (e.g., weather, news), prefer mcp.web_search / mcp.web_parser instead of declining.\n"
            "For greetings, small talk, or questions about who you are, answer directly with stop=true and final_answer; do NOT call rag.qa.\n"
            "If you can answer now, set stop=true and put the answer in final_answer.\n"
            "Never invent citations.\n"
        )

        if avatar_patch:
            system += "\n\n# AVATAR Policy Patch (deployment)\n" + avatar_patch.strip() + "\n"

        user = (
            f"User question:\n{user_question}\n\n"
            f"Available tools:\n{chr(10).join(tool_lines)}\n\n"
            f"Current memory (tool results so far):\n{memory_brief}\n\n"
            "Output JSON schema (strict):\n"
            "{\n"
            '  "tool": "<tool name or final>",\n'
            '  "args": { ... },\n'
            '  "stop": false,\n'
            '  "final_answer": "<string; required when stop=true>",\n'
            '  "thought": "<optional brief rationale>"\n'
            "}\n"
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def plan_next(
        self,
        user_question: str,
        tools: List[Dict[str, Any]],
        memory: Dict[str, Any],
    ) -> Tuple[PlanStep, Dict[str, Any]]:
        """
        Returns (plan_step, usage_dict).
        Falls back to rag.qa if parsing fails.
        """
        messages = self._build_messages(user_question, tools, memory)

        # Try to enable "thinking" if your LLMClient supports it; otherwise ignore.
        try:
            text, usage = self.llm.chat(messages, enable_thinking=True)
        except TypeError:
            text, usage = self.llm.chat(messages)

        obj = _try_parse_json(text)
        allowed = {t["name"] for t in tools} | {"final"}

        if not obj or "tool" not in obj:
            if _is_small_talk(user_question):
                return PlanStep(tool="final", args={}, stop=True, final_answer=_default_identity_answer(), thought="small_talk"), usage
            return PlanStep(tool="final", args={}, stop=True, final_answer="I had trouble planning an action. Please rephrase the request.", thought="fallback_parse_failed"), usage

        tool = str(obj.get("tool"))
        args = obj.get("args") or {}
        stop = bool(obj.get("stop", False))
        final_answer = obj.get("final_answer", None)
        thought = obj.get("thought")

        if tool not in allowed:
            if _is_small_talk(user_question):
                return PlanStep(tool="final", args={}, stop=True, final_answer=_default_identity_answer(), thought="small_talk_invalid_tool"), usage
            return PlanStep(tool="final", args={}, stop=True, final_answer="I cannot use that action. Please rephrase your request.", thought="fallback_invalid_tool"), usage

        if stop:
            # If the model forgot to provide a final_answer, use identity for small talk, otherwise a polite default.
            fallback_answer = _default_identity_answer() if _is_small_talk(user_question) else "I can answer directly: no tool call needed."
            return PlanStep(tool="final", args={}, stop=True, final_answer=str(final_answer or fallback_answer), thought=thought), usage

        return PlanStep(tool=tool, args=dict(args), stop=False, thought=thought), usage
