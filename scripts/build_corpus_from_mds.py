import os
import glob
import json
import sys
from datetime import datetime

# Ensure the project root is in sys.path so imports work when running from scripts/.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.ingest.md_loader import load_md
from rag.ingest.chunker_md import chunk_markdown
from obs.logger import JsonLogger, new_run_id


def main() -> None:
    """
    Build a JSONL corpus from Markdown files under data/mds/.

    Outputs:
      - data/corpus/corpus.jsonl: one JSON record per chunk
      - data/corpus/manifest.json: reproducibility metadata (params, stats, inputs)
      - logs/run_<run_id>.jsonl: structured event logs
    """
    md_paths = sorted(glob.glob("data/mds/**/*.md", recursive=True))
    if not md_paths:
        print("No Markdown files found under data/mds/**/*.md")
        return

    os.makedirs("data/corpus", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    out_jsonl = "data/corpus/corpus.jsonl"
    manifest_path = "data/corpus/manifest.json"

    # Chunking parameters (tune later based on retrieval quality and latency).
    chunk_size = 1200
    overlap = 200

    # Structured logger (JSONL events).
    run_id = new_run_id()
    logger = JsonLogger(log_path=f"logs/run_{run_id}.jsonl", run_id=run_id)

    logger.emit(
        event="corpus.build.start",
        stage="ingest",
        attrs={
            "num_mds": len(md_paths),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "out_jsonl": out_jsonl,
        },
    )

    total_chunks = 0
    inputs = []

    # Write corpus.jsonl (one record per chunk).
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for path in md_paths:
            logger.emit(event="md.load.start", stage="extract", attrs={"md": path})
            doc = load_md(path)
            logger.emit(
                event="md.load.end",
                stage="extract",
                attrs={"md": path, "chars": len(doc.text)},
            )

            chunks = chunk_markdown(
                source_path=doc.source_path,
                title=doc.title,
                md_text=doc.text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            logger.emit(
                event="chunk.create.end",
                stage="chunk",
                attrs={"md": path, "chunks": len(chunks)},
            )

            inputs.append({"path": path, "chars": len(doc.text), "chunks": len(chunks)})

            for c in chunks:
                record = {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "metadata": c.metadata,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            print(f"[OK] {path} -> chunks={len(chunks)}")

    # Persist manifest for reproducibility.
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "params": {"chunk_size": chunk_size, "overlap": overlap},
        "stats": {"mds": len(md_paths), "chunks": total_chunks},
        "inputs": inputs,
        "output": {"corpus_jsonl": out_jsonl},
        "log": {"run_jsonl": f"logs/run_{run_id}.jsonl"},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.emit(
        event="corpus.build.end",
        stage="ingest",
        attrs={"mds": len(md_paths), "chunks": total_chunks, "manifest": manifest_path},
    )

    print(f"Wrote corpus: {out_jsonl}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote logs: logs/run_{run_id}.jsonl")


if __name__ == "__main__":
    main()
