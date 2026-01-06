import os, sys

# Make sure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retrieve import bm25

def main():
    corpus = "data/corpus/corpus.jsonl"
    index_path = "data/index/bm25.pkl"

    bm25.build(corpus, index_path)
    store = bm25.load(index_path)

    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break
        hits = bm25.search(store, q, k=5)
        for rank, h in enumerate(hits, start=1):
            md = h["metadata"]
            print(f"\n[{rank}] score={h['score']:.3f} chunk_id={h['chunk_id']}")
            print(f"    section={md.get('section')} | lines={md.get('line_start')}-{md.get('line_end')} | source={md.get('source_path')}")
            print(h["text"][:220].replace("\n", " ") + ("..." if len(h["text"]) > 220 else ""))

if __name__ == "__main__":
    main()
