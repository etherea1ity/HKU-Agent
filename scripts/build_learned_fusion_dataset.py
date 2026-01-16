"""Build weakly-supervised train/test data for learned fusion (Logistic Regression).

This project already has two retrieval runs:
- BM25 (rank-bm25 over corpus chunks)
- Semantic (SentenceTransformer + FAISS cosine)

We generate queries from the corpus itself (course code + a few templated intents).
Labels are weak supervision: a candidate is positive iff its doc_id matches the query's target doc.

Outputs:
- data/learned_fusion/train.jsonl
- data/learned_fusion/test.jsonl

Each line is one (query, candidate_chunk) example with features.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rag.config import RagConfig
from rag.pipeline import stores
from rag.retrieve import bm25 as bm25_mod
from rag.retrieve import semantic as semantic_mod


_COURSE_CODE_RE = re.compile(r"\bCOMP\d{4}[A-Z]?\b", re.IGNORECASE)


@dataclass(frozen=True)
class Example:
    query_id: str
    query: str
    target_doc_id: str

    chunk_id: str
    doc_id: str
    label: int

    features: Dict[str, float]
    meta: Dict[str, Any]


def _iter_corpus(corpus_jsonl: str) -> Iterable[Dict[str, Any]]:
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_course_code(rec: Dict[str, Any]) -> Optional[str]:
    # Prefer from text because it's reliable for these course docs.
    text = rec.get("text") or ""
    m = _COURSE_CODE_RE.search(text)
    if m:
        return m.group(0).upper()

    # Fallback: sometimes the title contains it.
    meta = rec.get("metadata") or {}
    title = str(meta.get("title") or "")
    m2 = _COURSE_CODE_RE.search(title)
    if m2:
        return m2.group(0).upper()

    return None


def _build_queries_for_doc(course_code: str) -> List[str]:
    # Simple templates; easy to extend later.
    base = course_code.upper()
    return [
        base,
        f"{base} assessment",
        f"{base} exam",
        f"{base} schedule",
    ]


def _make_union_features(
    bm25_hits: List[Dict[str, Any]],
    sem_hits: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Return per-chunk features from union of hits."""

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

    # Fill missing values consistently.
    for f in by_id.values():
        if "bm25_score" not in f:
            f["bm25_score"] = 0.0
            f["bm25_rank"] = 9999.0
            f["bm25_present"] = 0.0
        if "semantic_score" not in f:
            f["semantic_score"] = 0.0
            f["semantic_rank"] = 9999.0
            f["semantic_present"] = 0.0

        # Rank-based features (RRF-like), helpful for scale mismatch.
        f["bm25_rrf"] = 1.0 / (60.0 + f["bm25_rank"]) if f["bm25_present"] > 0 else 0.0
        f["semantic_rrf"] = 1.0 / (60.0 + f["semantic_rank"]) if f["semantic_present"] > 0 else 0.0
        f["best_rank"] = min(f["bm25_rank"], f["semantic_rank"])
        f["best_rrf"] = max(f["bm25_rrf"], f["semantic_rrf"])

    return by_id


def _pick_negatives(
    candidates: List[Example],
    neg_per_pos: int,
    rng: random.Random,
) -> List[Example]:
    pos = [e for e in candidates if e.label == 1]
    neg = [e for e in candidates if e.label == 0]
    if not pos:
        return []
    keep_neg = min(len(neg), neg_per_pos * len(pos))
    rng.shuffle(neg)
    return pos + neg[:keep_neg]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/learned_fusion")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--bm25-k", type=int, default=50)
    ap.add_argument("--sem-k", type=int, default=50)

    ap.add_argument("--neg-per-pos", type=int, default=6)
    ap.add_argument("--test-doc-ratio", type=float, default=0.25)

    args = ap.parse_args()

    rng = random.Random(args.seed)

    cfg = RagConfig()
    bm25_store = stores.load_bm25_store(cfg)
    sem_store = stores.load_semantic_store(cfg)

    # Group corpus records by doc_id and keep a representative record for query generation.
    doc_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for rec in _iter_corpus(cfg.corpus_jsonl):
        doc_to_records.setdefault(rec["doc_id"], []).append(rec)

    doc_ids = sorted(doc_to_records.keys())
    if not doc_ids:
        raise RuntimeError("No documents found in corpus")

    rng.shuffle(doc_ids)
    test_n = max(1, int(round(len(doc_ids) * float(args.test_doc_ratio))))
    test_docs = set(doc_ids[:test_n])

    train_path = os.path.join(args.out_dir, "train.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    counts = {"train": 0, "test": 0, "queries": 0, "skipped_no_pos": 0, "skipped_no_code": 0}

    def write_examples(path: str, examples: List[Example]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    train_examples: List[Example] = []
    test_examples: List[Example] = []

    for doc_id, records in doc_to_records.items():
        rep = records[0]
        course_code = _extract_course_code(rep)
        if not course_code:
            counts["skipped_no_code"] += 1
            continue

        queries = _build_queries_for_doc(course_code)

        for qi, q in enumerate(queries):
            query_id = f"{doc_id}:{qi}"

            bm25_hits = bm25_mod.search(bm25_store, q, k=args.bm25_k)
            sem_hits = semantic_mod.search(sem_store, q, k=args.sem_k, ef_search=cfg.sem_ef_search)

            features_by_chunk = _make_union_features(bm25_hits, sem_hits)

            # Need doc_id/text/metadata for each chunk_id. Best-effort: take it from whichever run contains it.
            payload_by_chunk: Dict[str, Dict[str, Any]] = {}
            for h in bm25_hits + sem_hits:
                payload_by_chunk.setdefault(h["chunk_id"], h)

            candidates: List[Example] = []
            for chunk_id, feats in features_by_chunk.items():
                payload = payload_by_chunk.get(chunk_id, {})
                cand_doc_id = str(payload.get("doc_id") or "")

                label = 1 if cand_doc_id == doc_id else 0

                meta = payload.get("metadata") or {}
                text = str(payload.get("text") or "")
                meta_out = {
                    "title": meta.get("title"),
                    "source_path": meta.get("source_path"),
                    "text_preview": text[:160],
                }

                candidates.append(
                    Example(
                        query_id=query_id,
                        query=q,
                        target_doc_id=doc_id,
                        chunk_id=chunk_id,
                        doc_id=cand_doc_id,
                        label=label,
                        features=feats,
                        meta=meta_out,
                    )
                )

            # Keep all positives + sampled negatives.
            kept = _pick_negatives(candidates, neg_per_pos=args.neg_per_pos, rng=rng)
            if not kept:
                counts["skipped_no_pos"] += 1
                continue

            rng.shuffle(kept)

            if doc_id in test_docs:
                test_examples.extend(kept)
                counts["test"] += len(kept)
            else:
                train_examples.extend(kept)
                counts["train"] += len(kept)

            counts["queries"] += 1

    write_examples(train_path, train_examples)
    write_examples(test_path, test_examples)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "out_dir": args.out_dir,
                "train_path": train_path,
                "test_path": test_path,
                "seed": args.seed,
                "bm25_k": args.bm25_k,
                "sem_k": args.sem_k,
                "neg_per_pos": args.neg_per_pos,
                "test_doc_ratio": args.test_doc_ratio,
                "test_docs": sorted(test_docs),
                "counts": counts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote: {train_path} ({len(train_examples)} examples)")
    print(f"Wrote: {test_path} ({len(test_examples)} examples)")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
