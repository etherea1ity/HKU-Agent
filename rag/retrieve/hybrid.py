from typing import Any, Dict, List, Optional, Tuple


def rrf_fuse(
    runs: Dict[str, List[Dict[str, Any]]],
    rrf_k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion.

    runs:
      {
        "bm25":    [ {"chunk_id":..., "score":..., "text":..., "metadata":...}, ... ],
        "semantic":[ {...}, ... ],
      }
    Each list must be ordered best->worst (rank 1 is first element).

    rrf_k:
      Larger => ranks contribute more smoothly; common defaults: 10 or 60.

    weights:
      Optional per-run weights, e.g. {"bm25": 1.0, "semantic": 1.2}

    Returns:
      List of fused hits, each contains:
        - chunk_id, fused_score
        - ranks: per-run rank
        - rrf: per-run contribution
        - plus merged fields (text/metadata) best-effort
    """
    weights = weights or {}

    # Build rank maps: run -> chunk_id -> rank (1-based)
    rank_maps: Dict[str, Dict[str, int]] = {}
    payload_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for run_name, hits in runs.items():
        rm: Dict[str, int] = {}
        pm: Dict[str, Dict[str, Any]] = {}
        for rank, h in enumerate(hits, start=1):
            cid = h.get("chunk_id")
            if not cid:
                raise ValueError(f"Missing chunk_id in run={run_name}")
            if cid not in rm:
                rm[cid] = rank
                pm[cid] = h
        rank_maps[run_name] = rm
        payload_maps[run_name] = pm

    # Union of all candidate ids
    all_ids = set()
    for rm in rank_maps.values():
        all_ids.update(rm.keys())

    fused: List[Tuple[str, float, Dict[str, int], Dict[str, float], Dict[str, Any]]] = []

    for cid in all_ids:
        fused_score = 0.0
        ranks: Dict[str, int] = {}
        contrib: Dict[str, float] = {}
        merged: Dict[str, Any] = {"chunk_id": cid}

        # Best-effort: keep a representative text/metadata from whichever run provides it first
        rep_text = None
        rep_meta = None

        for run_name, rm in rank_maps.items():
            if cid not in rm:
                continue
            r = rm[cid]
            w = float(weights.get(run_name, 1.0))
            c = w * (1.0 / (rrf_k + r))
            fused_score += c
            ranks[run_name] = r
            contrib[run_name] = c

            payload = payload_maps[run_name].get(cid, {})
            if rep_text is None and "text" in payload:
                rep_text = payload.get("text")
            if rep_meta is None and "metadata" in payload:
                rep_meta = payload.get("metadata")

        if rep_text is not None:
            merged["text"] = rep_text
        if rep_meta is not None:
            merged["metadata"] = rep_meta

        merged["fused_score"] = fused_score
        merged["ranks"] = ranks
        merged["rrf"] = contrib

        fused.append((cid, fused_score, ranks, contrib, merged))

    # Sort: higher fused_score first; tie-break by best (smallest) rank across runs
    def sort_key(x):
        _, score, ranks, _, _ = x
        best_rank = min(ranks.values()) if ranks else 10**9
        return (-score, best_rank)

    fused.sort(key=sort_key)

    return [x[4] for x in fused[:top_k]]


def hybrid_search(
    bm25_module,
    bm25_store,
    semantic_module,
    semantic_store,
    query: str,
    top_k: int = 5,
    bm25_k: int = 20,
    sem_k: int = 20,
    rrf_k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    sem_ef_search: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: run bm25 + semantic, then fuse with RRF.

    We pass modules in so you can keep your current structure:
      - bm25_module is rag.retrieve.bm25
      - semantic_module is rag.retrieve.semantic
    """
    bm25_hits = bm25_module.search(bm25_store, query, k=bm25_k)
    sem_hits = semantic_module.search(semantic_store, query, k=sem_k, ef_search=sem_ef_search)

    fused = rrf_fuse(
        runs={"bm25": bm25_hits, "semantic": sem_hits},
        rrf_k=rrf_k,
        weights=weights,
        top_k=top_k,
    )

    # Optional: attach raw per-run top lists for debugging
    # (comment out if you want a cleaner output)
    for h in fused:
        h["_debug"] = {"bm25_top": bm25_hits[:5], "semantic_top": sem_hits[:5]}

    return fused
