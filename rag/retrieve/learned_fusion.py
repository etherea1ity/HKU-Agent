"""Learned fusion using a trained Logistic Regression (LR) model.

This module is intentionally small and side-effect free.
You can call it from rag/pipeline/retrieve.py later to replace/augment RRF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from functools import lru_cache


@dataclass(frozen=True)
class LearnedFusionResult:
    chunk_id: str
    score: float
    payload: Dict[str, Any]


def load_lr_artifact(path: str = "rag/models/learned_fusion_lr.joblib") -> Dict[str, Any]:
    return _load_lr_artifact_cached(path)


@lru_cache(maxsize=2)
def _load_lr_artifact_cached(path: str) -> Dict[str, Any]:
    return joblib.load(path)


def _features_from_runs(
    bm25_hits: List[Dict[str, Any]],
    sem_hits: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    by_id: Dict[str, Dict[str, float]] = {}

    for rank, h in enumerate(bm25_hits, start=1):
        cid = h["chunk_id"]
        f = by_id.setdefault(cid, {})
        f["bm25_score"] = float(h.get("score", 0.0))
        f["bm25_rank"] = float(rank)
        f["bm25_present"] = 1.0

    for rank, h in enumerate(sem_hits, start=1):
        cid = h["chunk_id"]
        f = by_id.setdefault(cid, {})
        f["semantic_score"] = float(h.get("score", 0.0))
        f["semantic_rank"] = float(rank)
        f["semantic_present"] = 1.0

    for f in by_id.values():
        if "bm25_score" not in f:
            f["bm25_score"] = 0.0
            f["bm25_rank"] = 9999.0
            f["bm25_present"] = 0.0
        if "semantic_score" not in f:
            f["semantic_score"] = 0.0
            f["semantic_rank"] = 9999.0
            f["semantic_present"] = 0.0

        f["bm25_rrf"] = 1.0 / (60.0 + f["bm25_rank"]) if f["bm25_present"] > 0 else 0.0
        f["semantic_rrf"] = 1.0 / (60.0 + f["semantic_rank"]) if f["semantic_present"] > 0 else 0.0
        f["best_rank"] = min(f["bm25_rank"], f["semantic_rank"])
        f["best_rrf"] = max(f["bm25_rrf"], f["semantic_rrf"])

    return by_id


def fuse_with_lr(
    query: str,
    bm25_hits: List[Dict[str, Any]],
    sem_hits: List[Dict[str, Any]],
    artifact: Optional[Dict[str, Any]] = None,
    model_path: str = "rag/models/learned_fusion_lr.joblib",
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Fuse candidates using a learned LR model.

    Returns a list of merged hit dicts (same style as existing retrieval code),
    sorted by predicted relevance probability.
    """

    artifact = artifact or load_lr_artifact(model_path)
    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    feats_by_id = _features_from_runs(bm25_hits, sem_hits)

    # Representative payload for each chunk_id
    payload_by_id: Dict[str, Dict[str, Any]] = {}
    for h in bm25_hits + sem_hits:
        payload_by_id.setdefault(h["chunk_id"], h)

    chunk_ids = list(feats_by_id.keys())
    x = np.zeros((len(chunk_ids), len(feature_names)), dtype=np.float32)

    for i, cid in enumerate(chunk_ids):
        feats = feats_by_id[cid]
        for j, name in enumerate(feature_names):
            x[i, j] = float(feats.get(name, 0.0))

    probs = pipeline.predict_proba(x)[:, 1]

    merged: List[Dict[str, Any]] = []
    for cid, p in zip(chunk_ids, probs.tolist()):
        payload = dict(payload_by_id.get(cid, {}))
        payload["chunk_id"] = cid
        payload["lf_score"] = float(p)
        payload["features"] = feats_by_id[cid]
        payload["query"] = query
        merged.append(payload)

    merged.sort(key=lambda h: h.get("lf_score", 0.0), reverse=True)
    return merged[:top_k]
