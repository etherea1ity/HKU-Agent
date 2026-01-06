import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NUM_IN_WORD_RE = re.compile(r"\d+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    """
    Tokenizer optimized for course docs:
    - basic \\w+ tokens
    - plus numeric substrings extracted from alphanumeric tokens (e.g., comp7103a -> 7103)
    """
    base = _WORD_RE.findall(text.lower())
    extra_nums: List[str] = []
    for tok in base:
        # if token mixes letters/digits, extract digit sequences
        if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
            extra_nums.extend(_NUM_IN_WORD_RE.findall(tok))
    return base + extra_nums

@dataclass
class BM25Store:
    """
    A lightweight container for:
      - BM25 model built over chunk texts
      - the original chunk records from corpus.jsonl
    """
    bm25: BM25Okapi
    chunks: List[Dict[str, Any]]

def build(corpus_jsonl: str, out_path: str) -> None:
    """
    Build BM25 index from corpus.jsonl and persist to out_path.

    Steps:
      1) read chunks from JSONL
      2) tokenize each chunk text
      3) build BM25 model
      4) save model + chunks as pickle
    """
    chunks: List[Dict[str, Any]] = []
    tokenized: List[List[str]] = []

    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(rec)
            tokenized.append(tokenize(rec["text"]))

    bm25 = BM25Okapi(tokenized)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

def load(path: str) -> BM25Store:
    """Load BM25 index from pickle."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return BM25Store(bm25=obj["bm25"], chunks=obj["chunks"])

def search(store: BM25Store, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search top-k chunks for query.

    Steps:
      1) tokenize query
      2) bm25 scores for all chunks
      3) take top-k by score
      4) return hits with score + chunk fields
    """
    qtok = tokenize(query)
    scores = store.bm25.get_scores(qtok)

    top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    hits: List[Dict[str, Any]] = []
    for i in top_ids:
        rec = store.chunks[i]
        hits.append({
            "chunk_id": rec["chunk_id"],
            "doc_id": rec["doc_id"],
            "text": rec["text"],
            "metadata": rec.get("metadata", {}),
            "score": float(scores[i]),
        })
    return hits
