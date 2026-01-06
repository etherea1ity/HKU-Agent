from typing import Any, Dict, List, Optional

from rag.config import RagConfig
from rag.pipeline.stores import load_bm25_store, load_semantic_store
from rag.retrieve import bm25, semantic, hybrid


from typing import Any, Dict, List

from rag.config import RagConfig
from rag.pipeline.stores import load_bm25_store, load_semantic_store
from rag.retrieve import bm25, semantic, hybrid

from rag.rerank.colbert_rerank import rerank_colbert


def retrieve(cfg: RagConfig, query: str) -> List[Dict[str, Any]]:
    bm25_store = load_bm25_store(cfg)
    sem_store = load_semantic_store(cfg)

    # 1) recall more candidates than final top_k if we plan to rerank
    candidate_k = cfg.rerank_candidates_k if cfg.use_colbert_rerank else cfg.top_k

    hits = hybrid.hybrid_search(
        bm25_module=bm25,
        bm25_store=bm25_store,
        semantic_module=semantic,
        semantic_store=sem_store,
        query=query,
        top_k=candidate_k,
        bm25_k=cfg.bm25_k,
        sem_k=cfg.sem_k,
        rrf_k=cfg.rrf_k,
        weights={"bm25": 1.0, "semantic": 1.0},
        sem_ef_search=cfg.sem_ef_search,
    )

    # 2) rerank (ColBERT-style late interaction)
    if cfg.use_colbert_rerank and hits:
        hits = rerank_colbert(
            query=query,
            hits=hits,
            model_name=cfg.colbert_model_name,
            query_max_length=cfg.colbert_query_max_length,
            doc_max_length=cfg.colbert_doc_max_length,
            normalize=cfg.colbert_normalize,
        )

    # 3) final top_k
    return hits[: cfg.top_k]

