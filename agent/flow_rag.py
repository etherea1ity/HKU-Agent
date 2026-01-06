import time
from dataclasses import dataclass
from typing import Any, Dict

from rag.config import RagConfig
from rag.pipeline.retrieve import retrieve
from rag.pipeline.context import build_context

from core.llm_client import LLMClient
from core.prompts import make_rewrite_messages, make_answer_messages


@dataclass
class RagFlow:
    cfg: RagConfig
    llm: LLMClient

    def run(self, user_question: str) -> Dict[str, Any]:
        """
        End-to-end:
          user_question
            -> LLM rewrite query
            -> RAG retrieve (hybrid)
            -> build context
            -> LLM answer
        """
        t0 = time.time()

        # 1) rewrite query (usually no need thinking)
        rewrite_msgs = make_rewrite_messages(user_question)
        rewritten_query, usage_rewrite = self.llm.chat(rewrite_msgs, enable_thinking=False)
        rewritten_query = rewritten_query.strip()

        # 2) retrieve
        hits = retrieve(self.cfg, rewritten_query)

        # 3) context
        context = build_context(self.cfg, hits)

        # 4) answer (can enable thinking if you want)
        ans_msgs = make_answer_messages(user_question, context)
        answer_text, usage_answer = self.llm.chat(ans_msgs, enable_thinking=True)

        t1 = time.time()

        return {
            "user_question": user_question,
            "rewritten_query": rewritten_query,
            "hits": hits,
            "context": context,
            "answer": answer_text,
            "usage": {
                "rewrite": usage_rewrite,
                "answer": usage_answer,
            },
            "latency_ms": int((t1 - t0) * 1000),
        }
