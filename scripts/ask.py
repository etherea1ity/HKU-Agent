import os
import sys

# Ensure project root import works when running "python scripts/ask.py"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import RagConfig
from rag.pipeline.answer import answer


def main():
    cfg = RagConfig()

    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            break

        out = answer(cfg, q)
        hits = out["hits"]

        print("\n===== RETRIEVAL (top hits) =====")
        for i, h in enumerate(hits, start=1):
            md = h.get("metadata", {}) or {}
            print(f"\n[{i}] fused={h.get('fused_score', 0):.6f} chunk_id={h.get('chunk_id')}")
            print(f"    ranks={h.get('ranks')} source={md.get('source_path')} section={md.get('section')}")
            txt = (h.get("text") or "").replace("\n", " ")
            print(txt[:200] + ("..." if len(txt) > 200 else ""))

        print("\n===== PROMPT PREVIEW =====\n")
        print(out["prompt"])
        print("\n=========================\n")


if __name__ == "__main__":
    main()
