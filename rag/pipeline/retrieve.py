from typing import Any, Dict, List
import re

from rag.config import RagConfig
from rag.pipeline.stores import load_bm25_store, load_semantic_store
from rag.retrieve import bm25, semantic, hybrid
from rag.rerank.colbert_rerank import rerank_colbert


def _pick_score(hit: Dict[str, Any]) -> float:
    """
    Pick a stable score field for thresholding.
    Priority order tries to use the most "final" score if present.
    """
    for key in ("colbert_score", "fused_score", "score", "bm25_score", "semantic_score"):
        v = hit.get(key, None)
        if isinstance(v, (int, float)):
            return float(v)
    return float("-inf")


def _apply_threshold(
    hits: List[Dict[str, Any]],
    *,
    ratio: float,
    abs_threshold: float,
    max_keep: int,
    min_keep: int,
) -> List[Dict[str, Any]]:
    """
    Keep hits whose score >= max(best_score * ratio, abs_threshold).
    Fallback to min_keep if threshold filters too aggressively.
    """
    if not hits:
        return hits

    best = _pick_score(hits[0])
    for h in hits[1:]:
        s = _pick_score(h)
        if s > best:
            best = s

    thr = max(best * ratio, abs_threshold)
    kept = [h for h in hits if _pick_score(h) >= thr]

    if len(kept) < min_keep:
        kept = hits[:min_keep]

    if max_keep > 0 and len(kept) > max_keep:
        kept = kept[:max_keep]

    return kept


def _looks_like_course_query(q: str) -> bool:
    """
    Heuristic: course code / course number query tends to want broader recall.
    Examples: "COMP7107", "7107", "comp 7107".
    """
    q = q.strip()
    if re.search(r"\bcomp\s*\d{4}\b", q, re.IGNORECASE):
        return True
    if re.search(r"\b\d{4}\b", q):
        return True
    if re.search(r"\bcourse(s)?\b", q, re.IGNORECASE):
        return True
    return False


def retrieve(cfg: RagConfig, query: str, *, use_colbert: bool = True) -> List[Dict[str, Any]]:
    """
    Retrieve chunks for a query. If use_colbert is provided, it overrides cfg.use_colbert_rerank.
    """

    bm25_store = load_bm25_store(cfg)
    sem_store = load_semantic_store(cfg)

    # Final number of chunks returned for context building (fallback to cfg.top_k)
    final_k = getattr(cfg, "final_k", getattr(cfg, "top_k", 8))

    # First-stage recall size (fallback to old rerank_candidates_k or a reasonable default)
    recall_k = getattr(cfg, "recall_k", getattr(cfg, "rerank_candidates_k", max(final_k, 80)))

    # Threshold parameters (provide safe defaults if not configured)
    ratio = getattr(cfg, "score_ratio", 0.65)
    abs_thr = getattr(cfg, "score_abs_threshold", float("-inf"))
    min_keep = getattr(cfg, "min_keep", min(5, final_k))
    max_keep = getattr(cfg, "max_keep", final_k)

    # Course-like queries usually want broader coverage (more recall, lower ratio, higher cap)
    if _looks_like_course_query(query):
        recall_k = max(recall_k, 200)
        max_keep = max(max_keep, 40)
        ratio = min(ratio, 0.55)

    if getattr(cfg, "debug_retrieve", False):
        try:
            bm25_hits = bm25.search(bm25_store, query=query, top_k=cfg.bm25_k)
            sem_hits = semantic.search(sem_store, query=query, top_k=cfg.sem_k, ef_search=cfg.sem_ef_search)
            print(f"[debug] pre-hybrid bm25_hits={len(bm25_hits)} sem_hits={len(sem_hits)}")
        except Exception as e:
            print(f"[debug] pre-hybrid direct search failed: {e}")
    # 1) First-stage recall using hybrid search
    hits = hybrid.hybrid_search(
        bm25_module=bm25,
        bm25_store=bm25_store,
        semantic_module=semantic,
        semantic_store=sem_store,
        query=query,
        top_k=recall_k,
        bm25_k=cfg.bm25_k,
        sem_k=cfg.sem_k,
        rrf_k=cfg.rrf_k,
        weights={"bm25": 1.0, "semantic": 1.0},
        sem_ef_search=cfg.sem_ef_search,
        fusion_mode=getattr(cfg, "fusion_mode", "rrf"),
        lr_model_path=getattr(cfg, "learned_fusion_model_path", "rag/models/learned_fusion_lr.joblib"),
    )
    if getattr(cfg, "debug_retrieve", False):
        try:
            bm25_hits = bm25.search(bm25_store, query=query, top_k=cfg.bm25_k)
            sem_hits = semantic.search(sem_store, query=query, top_k=cfg.sem_k, ef_search=cfg.sem_ef_search)
            print(f"[debug] post-hybrid bm25_hits={len(bm25_hits)} sem_hits={len(sem_hits)} fused={len(hits)}")
        except Exception as e:
            print(f"[debug] post-hybrid direct search failed: {e}")

    # 2) Optional rerank (ColBERT). Rerank only a prefix to control latency.
    # Keep tail to preserve recall coverage (do not shrink results just because we rerank).
    effective_colbert = cfg.use_colbert_rerank if use_colbert is None else bool(use_colbert)
    if effective_colbert and hits:
        rerank_k = getattr(cfg, "rerank_k", min(len(hits), getattr(cfg, "rerank_candidates_k", 30)))
        rerank_k = max(1, min(rerank_k, len(hits)))

        head = hits[:rerank_k]
        tail = hits[rerank_k:]

        head = rerank_colbert(
            query=query,
            hits=head,
            model_name=cfg.colbert_model_name,
            query_max_length=cfg.colbert_query_max_length,
            doc_max_length=cfg.colbert_doc_max_length,
            normalize=cfg.colbert_normalize,
        )

        hits = head + tail

    # 3) Threshold filtering + final cap
    hits = _apply_threshold(
        hits,
        ratio=ratio,
        abs_threshold=abs_thr,
        max_keep=max_keep,
        min_keep=min_keep,
    )

    # 4) Backward compatibility: ensure we never return more than final_k unless max_keep says otherwise
    # If you want max_keep to fully control output size, you can remove this line.
    if final_k > 0 and len(hits) > final_k and max_keep == final_k:
        hits = hits[:final_k]

    return hits
