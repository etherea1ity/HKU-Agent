from typing import Any, Dict

from rag.config import RagConfig
from rag.pipeline.query_rewrite import rewrite_query
from rag.pipeline.retrieve import retrieve
from rag.pipeline.context import build_context
from rag.pipeline.prompt import make_prompt


def answer(cfg: RagConfig, user_question: str) -> Dict[str, Any]:
    """
    Minimal RAG pipeline (no LLM call yet).
    Returns:
      - query: rewritten query used for retrieval
      - hits: retrieved chunks
      - context: formatted context string
      - prompt: final prompt to send to an LLM
    """
    query = rewrite_query(user_question)
    hits = retrieve(cfg, query)
    context = build_context(cfg, hits)
    prompt = make_prompt(cfg, user_question, context)

    return {
        "query": query,
        "hits": hits,
        "context": context,
        "prompt": prompt,
    }
