import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retrieve import bm25, semantic, hybrid

def main():
    corpus = "data/corpus/corpus.jsonl"

    bm25_index = "data/index/bm25.pkl"
    sem_dir = "data/index/semantic"

    # Build/load BM25
    bm25.build(corpus, bm25_index)
    bm25_store = bm25.load(bm25_index)

    # Build/load semantic (assumes your semantic.py supports index_type="hnsw")
    semantic.build(corpus, sem_dir, index_type="hnsw")
    sem_store = semantic.load(sem_dir, cache_model=True)

    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break

        hits = hybrid.hybrid_search(
            bm25_module=bm25,
            bm25_store=bm25_store,
            semantic_module=semantic,
            semantic_store=sem_store,
            query=q,
            top_k=5,
            bm25_k=20,
            sem_k=20,
            rrf_k=60,
            weights={"bm25": 1.0, "semantic": 1.0},
            sem_ef_search=128,  # you can tune this
        )

        for rank, h in enumerate(hits, start=1):
            md = h.get("metadata", {}) or {}
            print(f"\n[{rank}] fused={h['fused_score']:.6f} chunk_id={h['chunk_id']}")
            print(f"    ranks={h['ranks']} | source={md.get('source_path')} | section={md.get('section')}")
            txt = (h.get("text") or "").replace("\n", " ")
            print(txt[:240] + ("..." if len(txt) > 240 else ""))

if __name__ == "__main__":
    main()
