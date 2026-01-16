"""
Quick debug script to verify retrieval end-to-end without running FastAPI.
Run with project venv:
    python -m scripts.debug_retrieve "COMP 7107" "HKU courses"
If no args provided, uses a small default list.
"""
from pathlib import Path
import sys

from rag.config import RagConfig
from rag.pipeline.retrieve import retrieve
from rag.pipeline.context import build_context
from rag.pipeline.stores import load_bm25_store, load_semantic_store

def main(queries):
    cfg = RagConfig()
    print("cwd", Path.cwd())
    print("corpus", cfg.corpus_jsonl, "exists=", Path(cfg.corpus_jsonl).exists())

    bm25_store = load_bm25_store(cfg)
    sem_store = load_semantic_store(cfg)
    print("bm25 chunks", len(bm25_store.chunks))
    print("semantic items", len(sem_store.items))

    for q in queries:
        hits = retrieve(cfg, q, use_colbert=False)
        ctx = build_context(cfg, hits)
        print("\n=== Query:", q)
        print("hits=", len(hits))
        for h in hits[:3]:
            print("  -", h.get("doc_id"), h.get("chunk_id"), h.get("score"), h.get("text", "")[:80].replace("\n", " "))
        print("--- context preview ---")
        print(ctx[:500])

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["COMP 7107", "HKU courses", "COMP7103", "database course"]
    main(args)
