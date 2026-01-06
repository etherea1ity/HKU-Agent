import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization to unit vectors."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


@dataclass
class SemanticStore:
    # FAISS index (Flat or HNSW)
    index: faiss.Index
    # items[i] corresponds to vector i in the index
    items: List[Dict[str, Any]]
    # embedding model name used for both corpus and query
    model_name: str
    # embedding dimension
    dim: int
    # whether we normalized vectors (cosine via IP)
    normalized: bool
    # optional cached model for faster repeated queries
    _model: Optional[SentenceTransformer] = None


def _load_corpus_items(corpus_jsonl: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    items: List[Dict[str, Any]] = []
    texts: List[str] = []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            items.append({
                "chunk_id": rec["chunk_id"],
                "doc_id": rec["doc_id"],
                "text": text,
                "metadata": rec.get("metadata", {}),
            })
            texts.append(text)
    return items, texts


def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize ourselves for clarity
    )
    return emb.astype("float32")


def build(
    corpus_jsonl: str,
    out_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_type: str = "hnsw",  # "hnsw" or "flat"
    metric: str = "cosine",    # "cosine" (recommended). We implement cosine via normalize + IP.
    batch_size: int = 64,
    # HNSW params (only used when index_type="hnsw")
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> None:
    """
    Build a semantic vector index from corpus.jsonl.

    Files written to out_dir:
      - faiss.index    (FAISS index)
      - meta.pkl       (items list aligned to vectors)
      - config.pkl     (model_name, dim, normalized, index_type, metric, hnsw params)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load corpus
    items, texts = _load_corpus_items(corpus_jsonl)

    # 2) Embed all chunk texts
    model = SentenceTransformer(model_name)
    emb = _embed_texts(model, texts, batch_size=batch_size)

    # 3) Normalize if using cosine
    normalized = False
    if metric == "cosine":
        emb = l2_normalize(emb)
        normalized = True
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    else:
        # If you want L2 metric in the future, set metric="l2" and do not normalize.
        faiss_metric = faiss.METRIC_L2

    dim = int(emb.shape[1])

    # 4) Create index
    if index_type == "flat":
        # Exact search
        if faiss_metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

    elif index_type == "hnsw":
        # ANN search using HNSW
        # Use metric=INNER_PRODUCT for cosine (with normalized vectors)
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss_metric)
        index.hnsw.efConstruction = int(ef_construction)
        index.hnsw.efSearch = int(ef_search)  # can also be set at query time
    else:
        raise ValueError(f"Unknown index_type: {index_type}. Use 'flat' or 'hnsw'.")

    # 5) Add vectors to index
    index.add(emb)

    # 6) Save artifacts
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(items, f)

    cfg = {
        "model_name": model_name,
        "dim": dim,
        "normalized": normalized,
        "index_type": index_type,
        "metric": metric,
        "hnsw": {
            "m": hnsw_m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
        },
    }
    with open(os.path.join(out_dir, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)


def load(out_dir: str, cache_model: bool = True) -> SemanticStore:
    """
    Load a semantic index from out_dir.
    Optionally cache the embedding model in memory for faster repeated queries.
    """
    index = faiss.read_index(os.path.join(out_dir, "faiss.index"))
    with open(os.path.join(out_dir, "meta.pkl"), "rb") as f:
        items = pickle.load(f)
    with open(os.path.join(out_dir, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)

    store = SemanticStore(
        index=index,
        items=items,
        model_name=cfg["model_name"],
        dim=int(cfg["dim"]),
        normalized=bool(cfg["normalized"]),
        _model=SentenceTransformer(cfg["model_name"]) if cache_model else None,
    )
    return store


def search(store: SemanticStore, query: str, k: int = 5, ef_search: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Search top-k semantic matches.

    ef_search:
      - only meaningful for HNSW
      - larger => better recall, slower
    """
    # For HNSW, allow per-query efSearch override
    if ef_search is not None and hasattr(store.index, "hnsw"):
        store.index.hnsw.efSearch = int(ef_search)

    model = store._model or SentenceTransformer(store.model_name)

    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
    if store.normalized:
        q = l2_normalize(q)

    scores, ids = store.index.search(q, k)  # shapes: (1,k), (1,k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    hits: List[Dict[str, Any]] = []
    for s, idx in zip(scores, ids):
        if idx < 0:
            continue
        rec = store.items[idx]
        hits.append({
            "chunk_id": rec["chunk_id"],
            "doc_id": rec["doc_id"],
            "text": rec["text"],
            "metadata": rec.get("metadata", {}),
            "score": float(s),  # cosine similarity if normalized+IP
        })
    return hits
