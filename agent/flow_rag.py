import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Optional

from rag.config import RagConfig
from rag.pipeline.retrieve import retrieve
from rag.pipeline.context import build_context, expand_hits
from core.llm_client import LLMClient
from core.prompts import make_rewrite_messages, make_answer_messages


@dataclass
class RagFlow:
    cfg: RagConfig
    llm: LLMClient

    def run_stream(
        self,
        user_question: str,
        use_colbert: bool = True,
        include_debug: bool = False,
        fusion_mode: Optional[str] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        Stream a full RAG run:
          1) rewrite (non-stream)
          2) retrieve + expand + build context (non-stream)
          3) answer generation (stream)
        """
        t0 = time.time()

        try:
            # 1) Rewrite
            rewrite_messages = make_rewrite_messages(user_question)
            rewritten_query, usage_rewrite = self.llm.chat(rewrite_messages)

            # 2) Retrieve
            cfg = self.cfg
            if fusion_mode:
                cfg = replace(self.cfg, fusion_mode=str(fusion_mode))

            hits = retrieve(cfg, rewritten_query, use_colbert=use_colbert)
            if not hits:
                hits = retrieve(cfg, user_question, use_colbert=use_colbert)

            hits = expand_hits(cfg, hits)
            context = build_context(cfg, hits)

            # Send meta first
            meta: Dict[str, Any] = {
                "type": "meta",
                "user_question": user_question,
                "rewritten_query": rewritten_query,
            }
            if include_debug:
                meta["hits"] = hits
                meta["context"] = context
            yield meta

            # 3) Answer (stream)
            answer_messages = make_answer_messages(user_question, context)
            answer_parts = []

            for ev in self.llm.chat_stream(answer_messages):
                if ev["type"] == "delta":
                    answer_parts.append(ev["text"])
                    yield {"type": "delta", "text": ev["text"]}
                elif ev["type"] == "usage":
                    yield {"type": "usage", "usage": ev.get("usage", {})}

            t1 = time.time()
            yield {
                "type": "done",
                "answer": "".join(answer_parts),
                "usage": {
                    "rewrite": usage_rewrite,
                },
                "latency_ms": int((t1 - t0) * 1000),
            }

        except Exception as e:
            yield {"type": "error", "message": str(e)}
