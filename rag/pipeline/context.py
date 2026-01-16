import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from rag.config import RagConfig


def _get_text(rec: Dict[str, Any]) -> str:
    for k in ("text", "content", "chunk", "passage"):
        v = rec.get(k)
        if isinstance(v, str):
            return v
    return ""


def _get_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    m = rec.get("meta")
    return m if isinstance(m, dict) else {}


def _get_group_key(rec: Dict[str, Any]) -> str:
    """
    Group key identifies which document/record a chunk belongs to.
    Try multiple common keys to be robust across pipelines.
    """
    for k in ("doc_id", "source_path", "path", "source", "file", "document_id"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    meta = _get_meta(rec)
    for k in ("doc_id", "source_path", "path", "source", "file", "document_id"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return "UNKNOWN_DOC"


def _get_chunk_idx(rec: Dict[str, Any]) -> Optional[int]:
    for k in ("chunk_idx", "chunk_index", "idx", "i", "position"):
        v = rec.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)

    meta = _get_meta(rec)
    for k in ("chunk_idx", "chunk_index", "idx", "i", "position"):
        v = meta.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)

    return None


def _get_section_key(rec: Dict[str, Any]) -> Optional[str]:
    """
    Section key is optional. If your chunker stores heading/section info,
    this enables "same section expansion".
    """
    for k in ("section_id", "section", "heading", "header_path", "title_path"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    meta = _get_meta(rec)
    for k in ("section_id", "section", "heading", "header_path", "title_path"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def _get_uid(rec: Dict[str, Any], group_key: str, chunk_idx: Optional[int]) -> str:
    for k in ("chunk_id", "id", "uid"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, int):
            return str(v)

    meta = _get_meta(rec)
    for k in ("chunk_id", "id", "uid"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, int):
            return str(v)

    if chunk_idx is not None:
        return f"{group_key}::#{chunk_idx}"
    return f"{group_key}::#{hash(_get_text(rec))}"


@dataclass
class CorpusIndex:
    by_group: Dict[str, List[Dict[str, Any]]]
    by_uid: Dict[str, Dict[str, Any]]
    by_section: Dict[Tuple[str, str], List[Dict[str, Any]]]


@lru_cache(maxsize=1)
def _load_corpus_index(corpus_jsonl_path: str) -> CorpusIndex:
    """
    Load corpus.jsonl once per process and build lookup indexes.
    """
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    by_uid: Dict[str, Dict[str, Any]] = {}
    by_section: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            group_key = _get_group_key(rec)
            chunk_idx = _get_chunk_idx(rec)
            section_key = _get_section_key(rec)
            uid = _get_uid(rec, group_key, chunk_idx)

            # Normalize fields for later use
            rec["_group_key"] = group_key
            rec["_chunk_idx"] = chunk_idx
            rec["_section_key"] = section_key
            rec["_uid"] = uid

            by_uid[uid] = rec

            by_group.setdefault(group_key, []).append(rec)

            if section_key:
                by_section.setdefault((group_key, section_key), []).append(rec)

    # Sort group chunks by chunk_idx if present; fallback keeps file order
    for g, lst in by_group.items():
        lst.sort(key=lambda x: (x.get("_chunk_idx") is None, x.get("_chunk_idx", 10**9)))

    for k, lst in by_section.items():
        lst.sort(key=lambda x: (x.get("_chunk_idx") is None, x.get("_chunk_idx", 10**9)))

    return CorpusIndex(by_group=by_group, by_uid=by_uid, by_section=by_section)


def expand_hits(cfg: RagConfig, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand seed hits by adding neighboring chunks (and optionally same-section chunks).
    This prevents missing relevant content within the same document/course file.
    """
    if not hits:
        return hits

    corpus = _load_corpus_index(cfg.corpus_jsonl)

    window = int(getattr(cfg, "expand_neighbor_window", 1))
    use_section = bool(getattr(cfg, "expand_same_section", True))

    # Preserve group order by first appearance in seed hits
    group_order: List[str] = []
    group_seen = set()

    # Collect desired uids to keep
    keep_uids = set()

    # Also keep a mapping to preserve original seed scores if possible
    seed_score: Dict[str, float] = {}

    def pick_score(h: Dict[str, Any]) -> Optional[float]:
        for key in ("colbert_score", "fused_score", "score", "bm25_score", "semantic_score"):
            v = h.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        return None

    for h in hits:
        g = _get_group_key(h)
        idx = _get_chunk_idx(h)
        sec = _get_section_key(h)
        uid = _get_uid(h, g, idx)

        if g not in group_seen:
            group_seen.add(g)
            group_order.append(g)

        sc = pick_score(h)
        if sc is not None:
            seed_score[uid] = sc

        # Keep the seed itself if it exists in corpus index; otherwise keep by uid anyway
        keep_uids.add(uid)

        # Neighbor expansion within the same group
        if idx is not None and g in corpus.by_group:
            for delta in range(-window, window + 1):
                j = idx + delta
                if j < 0:
                    continue
                # Find the chunk in the group with _chunk_idx == j
                # Since group list is sorted, we do a simple scan (group size is usually small).
                for rec in corpus.by_group[g]:
                    if rec.get("_chunk_idx") == j:
                        keep_uids.add(rec.get("_uid"))
                        break

        # Same-section expansion (optional)
        if use_section and sec and (g, sec) in corpus.by_section:
            for rec in corpus.by_section[(g, sec)]:
                keep_uids.add(rec.get("_uid"))

    # Build expanded list in stable order: group order, then chunk_idx
    expanded: List[Dict[str, Any]] = []
    for g in group_order:
        group_recs = corpus.by_group.get(g, [])
        # Keep only the selected uids in this group
        selected = [r for r in group_recs if r.get("_uid") in keep_uids]
        selected.sort(key=lambda x: (x.get("_chunk_idx") is None, x.get("_chunk_idx", 10**9)))

        for r in selected:
            uid = r.get("_uid")
            out = dict(r)

            # Ensure downstream always has a "text" field
            if "text" not in out and "content" in out:
                out["text"] = out["content"]
            if "text" not in out:
                out["text"] = _get_text(out)

            # Attach seed score if this chunk was a seed, otherwise inherit 0.0
            # This is only for debugging; reranking/thresholding already happened upstream.
            if "score" not in out:
                if uid in seed_score:
                    out["score"] = seed_score[uid]
                else:
                    out["score"] = 0.0

            expanded.append(out)

    return expanded


def build_context(cfg: RagConfig, hits: List[Dict[str, Any]]) -> str:
    """
    Build a formatted context string from hits.
    Assumes hits are already expanded if desired.
    """
    # Optional: cap context length
    max_chars = int(getattr(cfg, "context_max_chars", 12000))

    blocks: List[str] = []
    total = 0

    for i, h in enumerate(hits, start=1):
        text = _get_text(h)
        if not text:
            continue

        group_key = h.get("_group_key") or _get_group_key(h)
        chunk_idx = h.get("_chunk_idx")
        if chunk_idx is None:
            chunk_idx = _get_chunk_idx(h)

        header = f"[{i}] {group_key}"
        if chunk_idx is not None:
            header += f" (chunk={chunk_idx})"

        block = header + "\n" + text.strip()

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)
