from typing import Any, Dict, List

from rag.config import RagConfig


def build_context(cfg: RagConfig, hits: List[Dict[str, Any]]) -> str:
    """
    Build a context string with citations.
    We keep it deterministic and traceable:
      [i] source, section, line range, chunk_id + text
    """
    blocks = []
    total = 0
    used = 0

    for i, h in enumerate(hits, start=1):
        if used >= cfg.max_chunks_in_context:
            break

        md = h.get("metadata", {}) or {}
        src = md.get("source_path", "")
        sec = md.get("section", "")
        ls = md.get("line_start", "")
        le = md.get("line_end", "")
        cid = h.get("chunk_id", "")

        txt = (h.get("text") or "").strip()
        if len(txt) > cfg.max_chars_per_chunk:
            txt = txt[: cfg.max_chars_per_chunk] + " ..."

        block = (
            f"[{i}]\n"
            f"source: {src}\n"
            f"section: {sec}\n"
            f"lines: {ls}-{le}\n"
            f"chunk_id: {cid}\n"
            f"text:\n{txt}\n"
        )

        # total context length control
        if total + len(block) > cfg.max_total_context_chars and used > 0:
            break

        blocks.append(block)
        total += len(block)
        used += 1

    return "\n---\n".join(blocks)
