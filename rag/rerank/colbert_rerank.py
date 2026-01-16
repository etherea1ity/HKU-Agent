from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download


# Avoid Transformers trying to import TF/Keras paths
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def _cache_dir() -> str:
    """Resolve HF cache dir (prefer project-local)."""
    for key in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.getenv(key)
        if v:
            return v
    return str(Path(__file__).resolve().parents[2] / "data" / "models")


@lru_cache(maxsize=2)
def _load_model_and_tokenizer(model_name: str):
    """Cached loader; prefer local cache and avoid re-downloading if present."""
    cache_dir = _cache_dir()
    local_only = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("HF_LOCAL_ONLY") == "1"

    # Ensure snapshot exists in cache; fallback to online if allowed
    allow_patterns = [
        "config.json",
        "pytorch_model.bin",
        "*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
    ]
    try:
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=local_only,
            allow_patterns=allow_patterns,
        )
    except Exception:
        if local_only:
            raise
        # Try online download into cache_dir
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            allow_patterns=allow_patterns,
        )

    tok = AutoTokenizer.from_pretrained(local_path, cache_dir=cache_dir, local_files_only=True)
    model = AutoModel.from_pretrained(local_path, cache_dir=cache_dir, local_files_only=True)
    model.eval()
    return tok, model


def _encode_tokens(
    tok,
    model,
    text: str,
    max_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      token_embeddings: (L, H)
      attn_mask: (L,)
    """
    batch = tok(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)
        # last_hidden_state: (1, L, H)
        x = out.last_hidden_state.squeeze(0)  # (L, H)

    attn = batch["attention_mask"].squeeze(0).to(torch.bool)  # (L,)
    return x, attn


def _colbert_score(
    q_tok_emb: torch.Tensor,
    q_mask: torch.Tensor,
    d_tok_emb: torch.Tensor,
    d_mask: torch.Tensor,
) -> float:
    """
    ColBERT-style MaxSim scoring:
      score = sum_i max_j dot(q_i, d_j)

    We assume embeddings are already L2-normalized so dot == cosine.
    """
    # Select valid tokens (exclude padding)
    q = q_tok_emb[q_mask]  # (Lq, H)
    d = d_tok_emb[d_mask]  # (Ld, H)

    if q.numel() == 0 or d.numel() == 0:
        return float("-inf")

    # Similarity matrix: (Lq, Ld)
    sim = torch.matmul(q, d.transpose(0, 1))

    # For each query token, take best matching doc token
    max_sim, _ = sim.max(dim=1)  # (Lq,)

    # Sum over query tokens
    return float(max_sim.sum().item())


def rerank_colbert(
    query: str,
    hits: List[Dict[str, Any]],
    *,
    model_name: str = "distilbert-base-uncased",
    query_max_length: int = 64,
    doc_max_length: int = 256,
    device: str | None = None,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    """
    Rerank a candidate list using ColBERT-style late interaction.

    Input hits format: each hit should contain:
      - "text": chunk text
      - "chunk_id": id
      - "metadata": dict
      - optional "fused_score" from hybrid

    Output: same list, sorted by "colbert_score" desc.
    Each hit will be annotated with:
      - hit["colbert_score"] = float
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok, model = _load_model_and_tokenizer(model_name)
    model = model.to(device)

    # Encode query tokens once
    q_emb, q_mask = _encode_tokens(tok, model, query, query_max_length, device)
    if normalize:
        q_emb = F.normalize(q_emb, p=2, dim=1)

    scored: List[Dict[str, Any]] = []
    for h in hits:
        txt = (h.get("text") or "").strip()
        if not txt:
            h["colbert_score"] = float("-inf")
            scored.append(h)
            continue

        d_emb, d_mask = _encode_tokens(tok, model, txt, doc_max_length, device)
        if normalize:
            d_emb = F.normalize(d_emb, p=2, dim=1)

        s = _colbert_score(q_emb, q_mask, d_emb, d_mask)
        h["colbert_score"] = s
        scored.append(h)

    # Sort by colbert_score primarily; fallback to fused_score if present
    scored.sort(
        key=lambda x: (x.get("colbert_score", float("-inf")), x.get("fused_score", 0.0)),
        reverse=True,
    )
    return scored
