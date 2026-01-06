from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re
import hashlib

# our chunk output structure
@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    # metadata should help us locate the incites
    # so we need the path, section, line_start_end
    metadata: Dict[str, Any]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")

# we give every document a stable doc_id
def _stable_doc_id(source_path: str, text: str) -> str:
    # we use source_path + length of doc to do a hash
    # risk: if length doen't change then will collapse
    raw = (source_path + "|" + str(len(text))).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]

# get every sections
def split_by_headings(md_text: str) -> List[Tuple[str, int, int]]:
    # we split the md by section
    # return [(section_title, start_line, end_line)]
    # start from 1-indexed
    lines = md_text.splitlines()
    headings = []

    # find all lines
    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # we keep the level of sections
            headings.append((i, level, title))
    
    # if no section
    if not headings:
        return [("Document", 1, len(lines))]
    
    sections = []
    for idx, (line_no, level, title) in enumerate(headings):
        start = line_no
        end = (headings[idx + 1][0] - 1) if idx + 1 < len(headings) else len(lines)
        sections.append((f"{'#'*level} {title}", start, end))

    return sections

def chunk_markdown(
    source_path: str,
    title: str,
    md_text: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> List[Chunk]:
    """
    we split the md to chunks

    input:
    source_path, title, md_text(document), chunk_size/overlap

    output:
    List[Chunk], with metadata
    """
    doc_id = _stable_doc_id(source_path, md_text)
    lines = md_text.splitlines()

    sections = split_by_headings(md_text)
    chunks: List[Chunk] = []
    chunk_idx = 0

    for sec_title, lstart, lend in sections:
        # get the original texts
        sec_text = "\n".join(lines[lstart-1:lend]).strip()
        if not sec_text:
            continue

        # the second chunk: by number of character
        start = 0
        n = len(sec_text)

        while start < n:
            end = min(n, start + chunk_size)
            piece = sec_text[start:end].strip()
            if not piece:
                break

            chunk_id = f"{doc_id}::c{chunk_idx:05d}"
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=piece,
                metadata={
                    "title": title,
                    "section": sec_title,
                    "source_path": source_path,
                    "line_start": lstart,
                    "line_end": lend,
                    "char_start": start,
                    "char_end": end,
                }
            ))

            chunk_idx += 1

            # overlap：下一个 chunk 从 end-overlap 开始
            start = end - overlap if end < n else n
            if start < 0:
                start = 0

    return chunks
    
    