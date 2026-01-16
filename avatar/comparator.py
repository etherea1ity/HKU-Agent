from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.llm_client import LLMClient


def _compact_trace(trace: List[Dict[str, Any]], max_items: int = 40) -> List[Dict[str, Any]]:
    """Compress trace to the most informative events for comparison."""
    keep_types = {"plan_update", "tool_start", "tool_end", "error", "done", "meta", "agent_meta"}
    out: List[Dict[str, Any]] = []
    for ev in trace or []:
        t = ev.get("type")
        if t in keep_types:
            # drop large fields
            slim = dict(ev)
            if t == "meta":
                # meta can contain hits/context; keep only rewritten query and a short hint
                slim = {
                    "type": "meta",
                    "rewritten_query": (ev.get("rewritten_query") or ""),
                    "has_context": bool(ev.get("context")),
                    "hits_n": len(ev.get("hits") or []) if isinstance(ev.get("hits"), list) else None,
                }
            out.append(slim)
        if len(out) >= max_items:
            break
    return out


def _summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    meta = run.get("meta") if isinstance(run.get("meta"), dict) else {}
    trace = run.get("trace") if isinstance(run.get("trace"), list) else []

    return {
        "query": run.get("query"),
        "metric": run.get("metric"),
        "citations_ok": run.get("citations_ok"),
        "answer_preview": (run.get("answer") or "")[:400],
        "rewritten_query": meta.get("rewritten_query"),
        "trace": _compact_trace(trace),
    }


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
class ComparatorOutput:
    planner_system_patch: str
    rationale: str
    signals: List[str]


def run_comparator(
    llm: LLMClient,
    *,
    positive: List[Dict[str, Any]],
    negative: List[Dict[str, Any]],
) -> Tuple[Optional[ComparatorOutput], Dict[str, Any]]:
    """Generate a generalizable planner instruction patch from pos/neg runs."""

    pos_s = [_summarize_run(r) for r in positive]
    neg_s = [_summarize_run(r) for r in negative]

    system = (
        "You are the Comparator in an AVATAR-style Actor–Comparator framework. "
        "Your job is to compare successful (positive) vs failed (negative) agent runs, "
        "find consistent action-selection differences, and output a GENERAL instruction patch "
        "to improve the planner. Return ONLY valid JSON."
    )

    user = (
        "We have positive and negative agent runs. Each run includes: query, metric, citations_ok, "
        "answer_preview, rewritten_query, and a compact trace of events (plan_update/tool_end/error).\n\n"
        "Positive runs:\n"
        f"{json.dumps(pos_s, ensure_ascii=False, indent=2)[:18000]}\n\n"
        "Negative runs:\n"
        f"{json.dumps(neg_s, ensure_ascii=False, indent=2)[:18000]}\n\n"
        "Task:\n"
        "1) Identify 3-7 generalizable rules to improve action selection (tool choice, avoiding redundant searches, using evidence already in memory).\n"
        "2) Produce a concise patch that can be appended to the planner SYSTEM prompt.\n"
        "3) Keep it tool-agnostic but consistent with this project: prefer rag.* for local course facts; mcp.* for fresh web facts; don't invent citations.\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "planner_system_patch": "...",\n'
        '  "rationale": "...",\n'
        '  "signals": ["...", "..."]\n'
        "}\n"
    )

    text, usage = llm.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    obj = _try_parse_json(text)
    if not obj:
        return None, usage

    patch = obj.get("planner_system_patch")
    if not isinstance(patch, str) or not patch.strip():
        return None, usage

    rationale = str(obj.get("rationale", ""))
    signals_raw = obj.get("signals")
    signals: List[str] = []
    if isinstance(signals_raw, list):
        signals = [str(x) for x in signals_raw[:20]]

    return ComparatorOutput(planner_system_patch=patch.strip(), rationale=rationale, signals=signals), usage
