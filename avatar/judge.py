from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from core.llm_client import LLMClient


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None

    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(s[l : r + 1])
        except Exception:
            return None

    return None


@dataclass(frozen=True)
class JudgeResult:
    correctness: float
    groundedness: float
    reasoning: str = ""


def judge_answer(
    llm: LLMClient,
    *,
    question: str,
    context: str,
    answer: str,
) -> Tuple[Optional[JudgeResult], Dict[str, Any]]:
    """LLM-based judge.

    Returns (JudgeResult|None, usage).
    Scores:
      - correctness: 0..1
      - groundedness: 0..1 (is answer supported by provided context)

    Output must be JSON.
    """

    system = (
        "You are a strict evaluator for a RAG assistant. "
        "You must judge ONLY using the provided context. "
        "If the answer contains claims not supported by context, reduce groundedness. "
        "Return ONLY valid JSON."
    )

    user = (
        "Evaluate the assistant answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Context (evidence blocks):\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "correctness": 0.0,\n'
        '  "groundedness": 0.0,\n'
        '  "reasoning": "short rationale"\n'
        "}\n"
        "Rules:\n"
        "- correctness=1 means the answer fully addresses the question correctly.\n"
        "- groundedness=1 means every important claim is supported by the context.\n"
        "- If context is insufficient, you may keep correctness moderate/low and groundedness low.\n"
    )

    text, usage = llm.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    obj = _try_parse_json(text)
    if not obj:
        return None, usage

    try:
        c = float(obj.get("correctness", 0.0))
        g = float(obj.get("groundedness", 0.0))
        r = str(obj.get("reasoning", ""))
        c = max(0.0, min(1.0, c))
        g = max(0.0, min(1.0, g))
        return JudgeResult(correctness=c, groundedness=g, reasoning=r), usage
    except Exception:
        return None, usage
