import os
from functools import lru_cache
from typing import Tuple

from rag.config import RagConfig
from rag.retrieve import bm25, semantic


def _exists_semantic_index(cfg: RagConfig) -> bool:
    return (
        os.path.exists(os.path.join(cfg.semantic_index_dir, "faiss.index"))
        and os.path.exists(os.path.join(cfg.semantic_index_dir, "meta.pkl"))
        and os.path.exists(os.path.join(cfg.semantic_index_dir, "config.pkl"))
    )


@lru_cache(maxsize=1)
def load_bm25_store(cfg: RagConfig):
    # Build if missing
    if not os.path.exists(cfg.bm25_index_path):
        os.makedirs(os.path.dirname(cfg.bm25_index_path), exist_ok=True)
        bm25.build(cfg.corpus_jsonl, cfg.bm25_index_path)
    return bm25.load(cfg.bm25_index_path)


@lru_cache(maxsize=1)
def load_semantic_store(cfg: RagConfig):
    # Build if missing
    if not _exists_semantic_index(cfg):
        os.makedirs(cfg.semantic_index_dir, exist_ok=True)
        semantic.build(
            corpus_jsonl=cfg.corpus_jsonl,
            out_dir=cfg.semantic_index_dir,
            model_name=cfg.embedding_model_name,
            index_type=cfg.semantic_index_type,
            hnsw_m=cfg.hnsw_m,
            ef_construction=cfg.ef_construction,
            ef_search=cfg.ef_search_default,
        )
    return semantic.load(cfg.semantic_index_dir, cache_model=True)
