"""Collect AVATAR-style actor rollouts and (optionally) judge them.

This script runs the full agent (Planner + Tools) on a batch of queries, logs:
- plan/tool trace events
- final answer
- (optional) debug meta: hits + context
- (optional) LLM judge scores (correctness/groundedness)

Output JSONL:
- logs/avatar_runs.jsonl

Usage examples:
- python scripts/avatar_collect_runs.py --n 20 --fusion-mode lr
- python scripts/avatar_collect_runs.py --queries data/avatar/queries.txt --judge

Note: Judging uses LLM API; it can be slow/costly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from obs.logger import JsonLogger, new_run_id
from rag.config import RagConfig
from agent.flow_rag import RagFlow
from core.llm_client import LLMClient, LLMConfig
from agent.tools.registry import ToolRegistry
from agent.tools.rag_tools import RagSearchTool, RagOpenTool, RagAnswerTool, RagQATool
from agent.tools.mcp_adapter import MCPConfig, register_mcp_tools, register_dashscope_tools
from agent.planner import LLMPlanner
from agent.runtime import AgentRuntime

from avatar.metrics import citations_ok as citations_ok_fn, combine_metrics
from avatar.judge import judge_answer


_COURSE_CODE_RE = re.compile(r"\bCOMP\d{4}[A-Z]?\b", re.IGNORECASE)


def _iter_corpus(corpus_jsonl: str) -> Iterable[Dict[str, Any]]:
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_course_code(rec: Dict[str, Any]) -> Optional[str]:
    text = rec.get("text") or ""
    m = _COURSE_CODE_RE.search(text)
    if m:
        return m.group(0).upper()

    meta = rec.get("metadata") or {}
    title = str(meta.get("title") or "")
    m2 = _COURSE_CODE_RE.search(title)
    if m2:
        return m2.group(0).upper()

    return None


def _default_queries(cfg: RagConfig, n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    seen = set()
    pool: List[str] = []

    for rec in _iter_corpus(cfg.corpus_jsonl):
        code = _extract_course_code(rec)
        if not code or code in seen:
            continue
        seen.add(code)
        pool.extend([
            code,
            f"{code} assessment",
            f"{code} exam",
            f"{code} schedule",
        ])

    rng.shuffle(pool)
    return pool[:n]


def _read_queries(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".jsonl":
        out = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("query") or obj.get("q") or obj.get("text")
                if q:
                    out.append(str(q).strip())
        return [q for q in out if q]

    # plain text: one query per line
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q:
                out.append(q)
    return out


def _init_agent(project_root: Path) -> AgentRuntime:
    cfg = RagConfig()
    llm = LLMClient(LLMConfig())
    flow = RagFlow(cfg=cfg, llm=llm)

    registry = ToolRegistry()
    registry.register(RagSearchTool(cfg=cfg))
    registry.register(RagOpenTool(project_root=project_root))
    registry.register(RagAnswerTool(llm=llm))
    registry.register(RagQATool(flow=flow))

    mcp_cfg = MCPConfig.from_env()
    register_mcp_tools(registry, mcp_cfg)
    register_dashscope_tools(registry)

    planner = LLMPlanner(llm=llm)
    return AgentRuntime(registry=registry, planner=planner, llm=llm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/avatar_runs.jsonl")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--queries", default=None, help="Path to queries.txt or queries.jsonl")
    ap.add_argument("--n", type=int, default=20, help="How many queries to run when --queries is not provided")

    ap.add_argument("--judge", action="store_true", help="Run LLM judge (costly)")

    ap.add_argument("--rag-enabled", action="store_true", default=True)
    ap.add_argument("--agent-enabled", action="store_true", default=True)
    ap.add_argument("--web-enabled", action="store_true", default=False)

    ap.add_argument("--use-colbert", action="store_true", default=False)
    ap.add_argument("--fusion-mode", default="rrf", choices=["rrf", "lr"])

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_id = new_run_id()
    logger = JsonLogger(args.out, run_id=run_id)

    cfg = RagConfig()
    agent = _init_agent(project_root)

    if args.queries:
        queries = _read_queries(args.queries)
    else:
        queries = _default_queries(cfg, n=int(args.n), seed=int(args.seed))

    rng = random.Random(args.seed)
    rng.shuffle(queries)

    logger.emit("avatar_collect_start", attrs={"n": len(queries), "judge": bool(args.judge)})

    for qi, q in enumerate(queries, start=1):
        t0 = time.time()
        # Force debug=True so we can capture hits/context in meta for judging.
        events: List[Dict[str, Any]] = []
        answer_parts: List[str] = []
        meta: Dict[str, Any] = {}

        for ev in agent.run_stream(
            q,
            session_id=f"avatar:{run_id}:{qi}",
            debug=True,
            use_colbert=bool(args.use_colbert),
            rag_enabled=bool(args.rag_enabled),
            agent_enabled=bool(args.agent_enabled),
            web_enabled=bool(args.web_enabled),
            fusion_mode=str(args.fusion_mode),
            max_steps=6,
        ):
            et = ev.get("type")
            if et == "delta":
                answer_parts.append(ev.get("text", ""))
            elif et == "meta":
                meta = dict(ev)
            events.append(ev)
            if et == "done":
                break

        answer = "".join(answer_parts).strip()
        latency_ms = int((time.time() - t0) * 1000)

        context = str(meta.get("context") or "")
        citations_ok_flag = citations_ok_fn(answer, context, require_at_least_one=True)

        judge_obj = None
        judge_usage: Dict[str, Any] = {}
        metric = 0.0

        if args.judge and context:
            jr, judge_usage = judge_answer(
                agent.llm,
                question=q,
                context=context,
                answer=answer,
            )
            if jr:
                judge_obj = asdict(jr)
                metric_res = combine_metrics(
                    citations_ok_flag=citations_ok_flag,
                    correctness=jr.correctness,
                    groundedness=jr.groundedness,
                )
                metric = metric_res.metric
            else:
                metric = 0.0

        rec = {
            "ts": time.time(),
            "run_id": run_id,
            "query_index": qi,
            "query": q,
            "config": {
                "rag_enabled": bool(args.rag_enabled),
                "agent_enabled": bool(args.agent_enabled),
                "web_enabled": bool(args.web_enabled),
                "use_colbert": bool(args.use_colbert),
                "fusion_mode": str(args.fusion_mode),
            },
            "latency_ms": latency_ms,
            "answer": answer,
            "meta": meta,
            "trace": events,
            "citations_ok": citations_ok_flag,
            "judge": judge_obj,
            "judge_usage": judge_usage,
            "metric": metric,
        }

        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.emit(
            "avatar_collect_item",
            attrs={
                "i": qi,
                "latency_ms": latency_ms,
                "citations_ok": citations_ok_flag,
                "metric": metric,
            },
        )

    logger.emit("avatar_collect_done")
    print(f"Wrote runs to: {args.out}")


if __name__ == "__main__":
    main()
